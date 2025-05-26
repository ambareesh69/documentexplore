// ui/story.js
// An advanced version providing better insights and interactive charts

import { html, render } from 'https://cdn.jsdelivr.net/npm/lit-html@2.2.0/+esm';
import * as d3 from 'https://cdn.jsdelivr.net/npm/d3@7.8.2/+esm';

// ========== 1. Load config & cluster names ==========

async function loadConfig() {
  try {
    const response = await fetch('docexplore.json');
    if (!response.ok) throw new Error("Failed to load config");
    return await response.json();
  } catch (err) {
    console.error(err);
    return null;
  }
}

async function loadClusterNames(clusterNamesFile) {
  try {
    const response = await fetch(clusterNamesFile);
    if (!response.ok) throw new Error("Failed to load cluster names");
    return await response.json();
  } catch (err) {
    console.error(err);
    return {};
  }
}

// ========== 2. Create the base layout ==========

function createBaseTemplate(config) {
  return html`
    <h1>${config.title}</h1>
    <p>${config.description}</p>

    <!-- Stats & cluster listing -->
    <section id="insights" style="margin-bottom: 2rem;">
      <h2>Key Insights</h2>
      <div id="stats"></div>
      <div id="cluster-list" style="margin-top:1rem;"></div>
    </section>

    <!-- Charts row -->
    <section id="charts" style="display:flex; flex-wrap:wrap; gap:40px;">
      <div id="bar-chart" style="flex:1 1 400px;"></div>
      <div id="donut-chart" style="flex:1 1 400px;"></div>
    </section>
  `;
}

// ========== 3. Data processing & chart rendering ==========

// We'll store the final cluster array globally so we can reference it in multiple charts.
let clusterData = [];  // array of {id, name, count}

// Merges embeddings data + clusterNames => clusterData array
async function loadAndMergeData(config, clusterNames) {
  // 1. Fetch embeddings data
  const response = await fetch(config.data);
  const embeddings = await response.json();

  // 2. Count how many chunks are in each cluster
  const clusterCount = {};
  embeddings.forEach(item => {
    const c = item.cluster;
    clusterCount[c] = (clusterCount[c] || 0) + 1;
  });

  // 3. Merge with clusterNames => build array of objects
  //    Example: { id: '0', name: 'Some cluster name', count: 15 }
  clusterData = Object.entries(clusterCount).map(([id, count]) => ({
    id,
    name: clusterNames[id] || `Cluster ${id}`,
    count
  }));

  // Sort by numeric cluster ID, if you like (or by count descending)
  clusterData.sort((a, b) => +a.id - +b.id);
}

// ========== 4. Render "Key Insights" ==========

function renderKeyInsights() {
  const statsEl = document.getElementById('stats');
  const totalClusters = clusterData.length;
  const totalChunks = d3.sum(clusterData, d => d.count);
  const avgChunks = totalChunks / totalClusters;

  // Find top cluster(s)
  const maxCount = d3.max(clusterData, d => d.count);
  const topClusters = clusterData.filter(d => d.count === maxCount);

  const template = html`
    <ul>
      <li><strong>Total Clusters:</strong> ${totalClusters}</li>
      <li><strong>Total Chunks:</strong> ${totalChunks}</li>
      <li><strong>Average Chunks per Cluster:</strong> ${avgChunks.toFixed(2)}</li>
      <li>
        <strong>Most Discussed:</strong>
        ${topClusters.map(c => html`${c.name} (${c.count} chunks)`).join(', ')}
      </li>
    </ul>
  `;
  render(template, statsEl);
}

// ========== 5. Render cluster listing with color-coded labels & interactivity ==========

function renderClusterListing() {
  const container = document.getElementById('cluster-list');
  container.innerHTML = ''; // clear old

  // Create a color scale
  const colorScale = d3.scaleOrdinal(d3.schemeCategory10)
    .domain(clusterData.map(d => d.id));

  // For each cluster, create a clickable div
  clusterData.forEach(d => {
    const div = document.createElement('div');
    div.className = 'cluster-item';
    div.style.borderLeft = `8px solid ${colorScale(d.id)}`;
    div.style.paddingLeft = '8px';
    div.style.margin = '4px 0';
    div.style.cursor = 'pointer';
    div.dataset.clusterId = d.id; // store the cluster id for click handlers

    div.innerHTML = `<strong>${d.name}</strong> <span style="float:right;">(${d.count} chunks)</span>`;
    container.appendChild(div);

    // On click, highlight the bar/donut slice
    div.addEventListener('click', () => {
      highlightCluster(d.id);
    });
    // On mouseover, also highlight
    div.addEventListener('mouseover', () => highlightCluster(d.id));
    div.addEventListener('mouseout', () => highlightCluster(null));
  });
}

// ========== 6. Render a bar chart with tooltips & interactive highlight ==========

let barSvg, xScale, yScale, colorScaleBar;
function renderBarChart() {
  // Basic dimensions
  const width = 400, height = 300;
  const margin = { top: 30, right: 20, bottom: 30, left: 40 };

  // Remove old svg if any
  d3.select('#bar-chart').select('svg').remove();

  barSvg = d3.select('#bar-chart')
    .append('svg')
    .attr('width', width)
    .attr('height', height);

  // Setup scales
  xScale = d3.scaleBand()
    .domain(clusterData.map(d => d.id))
    .range([margin.left, width - margin.right])
    .padding(0.1);

  const maxCount = d3.max(clusterData, d => d.count);
  yScale = d3.scaleLinear()
    .domain([0, maxCount]).nice()
    .range([height - margin.bottom, margin.top]);

  colorScaleBar = d3.scaleOrdinal(d3.schemeCategory10)
    .domain(clusterData.map(d => d.id));

  // Draw bars
  const bars = barSvg.selectAll('.bar')
    .data(clusterData)
    .join('rect')
      .attr('class', 'bar')
      .attr('x', d => xScale(d.id))
      .attr('y', d => yScale(d.count))
      .attr('width', xScale.bandwidth())
      .attr('height', d => yScale(0) - yScale(d.count))
      .attr('fill', d => colorScaleBar(d.id));

  // Axes
  barSvg.append('g')
    .attr('transform', `translate(0,${height - margin.bottom})`)
    .call(d3.axisBottom(xScale));

  barSvg.append('g')
    .attr('transform', `translate(${margin.left},0)`)
    .call(d3.axisLeft(yScale));

  // Add a tooltip
  const tooltip = d3.select('#bar-chart')
    .append('div')
    .style('position','absolute')
    .style('background','#fff')
    .style('border','1px solid #ccc')
    .style('padding','4px 8px')
    .style('border-radius','4px')
    .style('display','none');

  bars.on('mouseover', function (event, d) {
      highlightCluster(d.id);
      tooltip.html(`<strong>${d.name}</strong><br/>Count: ${d.count}`)
        .style('display','block');
    })
    .on('mousemove', function(event) {
      tooltip
        .style('left', (event.pageX + 10) + 'px')
        .style('top', (event.pageY - 20) + 'px');
    })
    .on('mouseout', function() {
      highlightCluster(null);
      tooltip.style('display','none');
    })
    .on('click', function(event, d) {
      highlightCluster(d.id);
    });
}

// ========== 7. Render a donut chart ==========

let donutSvg, arcGenerator, pieGenerator, colorScaleDonut;
function renderDonutChart() {
  // Dimensions
  const width = 300, height = 300, radius = Math.min(width, height)/2;
  // Remove old svg
  d3.select('#donut-chart').select('svg').remove();

  donutSvg = d3.select('#donut-chart')
    .append('svg')
    .attr('width', width)
    .attr('height', height)
    .append('g')
    .attr('transform', `translate(${width/2}, ${height/2})`);

  // Color scale
  colorScaleDonut = d3.scaleOrdinal(d3.schemeCategory10)
    .domain(clusterData.map(d => d.id));

  // Pie & arc generator
  pieGenerator = d3.pie().value(d => d.count);
  arcGenerator = d3.arc().innerRadius(radius*0.5).outerRadius(radius*0.9);

  // Convert clusterData to arcs
  const arcs = pieGenerator(clusterData);

  // Add arcs
  donutSvg.selectAll('path')
    .data(arcs)
    .join('path')
      .attr('d', arcGenerator)
      .attr('fill', d => colorScaleDonut(d.data.id))
      .attr('stroke', '#fff')
      .style('stroke-width', '1px')
      .on('mouseover', function(event, d) {
        highlightCluster(d.data.id);
      })
      .on('mouseout', function() {
        highlightCluster(null);
      })
      .on('click', function(event, d) {
        highlightCluster(d.data.id);
      });
  
  // Optional: add labels
  donutSvg.selectAll('text')
    .data(arcs)
    .join('text')
      .attr('transform', d => `translate(${arcGenerator.centroid(d)})`)
      .attr('text-anchor','middle')
      .attr('font-size','10px')
      .text(d => d.data.id);
}

// ========== 8. Interactive highlight logic ==========

function highlightCluster(clusterId) {
  // 1. For the cluster listing, highlight the item
  const items = document.querySelectorAll('.cluster-item');
  items.forEach(el => {
    if (clusterId && el.dataset.clusterId === clusterId) {
      el.style.backgroundColor = '#fffae0';
    } else {
      el.style.backgroundColor = '';
    }
  });

  // 2. For the bar chart, we can lighten other bars
  barSvg?.selectAll('.bar')
    .attr('fill', d => {
      if (!clusterId) return colorScaleBar(d.id);
      return (d.id === clusterId) ? colorScaleBar(d.id) : '#ccc';
    });

  // 3. For the donut chart, lighten other slices
  donutSvg?.selectAll('path')
    .attr('fill', d => {
      if (!clusterId) return colorScaleDonut(d.data.id);
      return (d.data.id === clusterId) ? colorScaleDonut(d.data.id) : '#ccc';
    });
}

// ========== 9. Main init function ==========

async function init() {
  const config = await loadConfig();
  if (!config) return;

  // Render the base layout
  render(createBaseTemplate(config), document.getElementById('app'));

  // Load cluster names
  const clusterNames = await loadClusterNames(config.clusterNames);

  // Merge cluster data (embeddings + clusterNames)
  await loadAndMergeData(config, clusterNames);

  // Render insights
  renderKeyInsights();

  // Render cluster listing
  renderClusterListing();

  // Render bar chart
  renderBarChart();

  // Render donut chart
  renderDonutChart();
}

// Start
init();
