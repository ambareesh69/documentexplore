import streamlit as st
import os
import json
import time
import subprocess
import shutil
import pandas as pd
import numpy as np
from collections import Counter, defaultdict
import plotly.graph_objects as go
import plotly.express as px
from sklearn.manifold import MDS

# ------------------ Helper Functions ------------------

def clear_folder(folder):
    """Delete all files and subfolders in a folder and recreate it."""
    try:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder, exist_ok=True)
        # st.success(f"Cleared folder: {folder}")
    except Exception as e:
        st.error(f"Failed to clear folder {folder}: {e}")
        st.stop()

def load_json(relative_path):
    """Load a JSON file relative to this file."""
    path = os.path.join(os.path.dirname(__file__), relative_path)
    if not os.path.exists(path):
        st.error(f"File not found: {path}")
        st.stop()
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_pipeline():
    """Run the backend pipeline scripts sequentially."""
    pipeline_scripts = [
        ("backend/data_extraction.py", "Extracting text from document..."),
        ("backend/text_chunking.py", "Chunking text..."),
        ("backend/embedding_generation.py", "Generating embeddings..."),
        ("backend/clustering.py", "Clustering embeddings..."),
        ("backend/topic_naming.py", "Naming topics..."),
        ("backend/similarity.py", "Computing similarities..."),
        ("backend/generate_json.py", "Generating document configuration..."),
    ]
    for script, desc in pipeline_scripts:
        with st.spinner(desc):
            result = subprocess.run(["python", script], capture_output=True, text=True)
            if result.returncode != 0:
                st.error(f"Error in {script}:\n{result.stderr}")
                st.stop()
            time.sleep(1)

def extract_keywords_per_topic(clustered_data, topics_dict, top_n=5):
    """Extract top keywords per topic based on word frequency in excerpts."""
    topic_texts = defaultdict(list)
    for item in clustered_data:
        cid = item.get("cluster")
        text = item.get("text", "").lower()
        if cid in topics_dict:
            topic_texts[cid].append(text)
    
    topic_keywords = {}
    for cid, texts in topic_texts.items():
        all_text = " ".join(texts)
        words = [word for word in all_text.split() if len(word) > 3 and word.isalpha()]
        word_counts = Counter(words)
        top_keywords = [word for word, count in word_counts.most_common(top_n)]
        topic_keywords[topics_dict[cid]] = top_keywords
    return topic_keywords

def extract_all_keywords(clustered_data, top_n=20):
    """Extract top keywords across all topics for the word cloud."""
    all_text = " ".join(item.get("text", "").lower() for item in clustered_data)
    words = [word for word in all_text.split() if len(word) > 3 and word.isalpha()]
    word_counts = Counter(words)
    return word_counts.most_common(top_n)

def compute_topic_overlap(clustered_data, topics_dict):
    """Compute topic overlap based on shared keywords (simplified)."""
    topic_texts = defaultdict(list)
    for item in clustered_data:
        cid = item.get("cluster")
        text = item.get("text", "").lower()
        if cid in topics_dict:
            topic_texts[cid].append(text)
    
    topic_words = {}
    for cid, texts in topic_texts.items():
        all_text = " ".join(texts)
        words = set([word for word in all_text.split() if len(word) > 3 and word.isalpha()])
        topic_words[cid] = words
    
    overlaps = []
    topic_names = list(topics_dict.items())
    for i in range(len(topic_names)):
        for j in range(i + 1, len(topic_names)):
            cid_i, name_i = topic_names[i]
            cid_j, name_j = topic_names[j]
            words_i = topic_words.get(cid_i, set())
            words_j = topic_words.get(cid_j, set())
            common = len(words_i.intersection(words_j))
            total = len(words_i.union(words_j))
            overlap = common / total if total > 0 else 0
            if overlap > 0.1:  # Threshold for meaningful overlap
                overlaps.append((name_i, name_j, round(overlap, 2)))
    return overlaps

def compute_topic_positions(clustered_data, topics_dict):
    """Compute 2D positions for topics using MDS based on average embeddings."""
    cluster_embeddings = defaultdict(list)
    for item in clustered_data:
        cid = item.get("cluster")
        emb = item.get("embedding")
        if emb and cid in topics_dict:
            cluster_embeddings[cid].append(np.array(emb))
    avg_embs = {cid: np.mean(embs, axis=0) for cid, embs in cluster_embeddings.items() if embs}
    
    emb_list = []
    topic_names = []
    for cid in topics_dict.keys():
        if cid in avg_embs:
            emb_list.append(avg_embs[cid])
            topic_names.append(topics_dict[cid])
    
    if not emb_list:
        return None, None
    
    emb_array = np.array(emb_list)
    n = len(emb_array)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            vecA, vecB = emb_array[i], emb_array[j]
            dot = np.dot(vecA, vecB)
            normA = np.linalg.norm(vecA)
            normB = np.linalg.norm(vecB)
            sim = dot / (normA * normB) if normA and normB else 0.0
            dist = 1 - sim  # Convert similarity to distance
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    positions = mds.fit_transform(dist_matrix)
    return positions, topic_names

# ------------------ Main App Logic ------------------

if "doc_uploaded" not in st.session_state or not st.session_state.doc_uploaded:
    # Upload Page
    st.title("Document Insights")
    st.markdown("Please upload a PDF document to begin analysis.")
    uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])
    
    if uploaded_pdf is not None:
        reports_folder = os.path.join(os.path.dirname(__file__), "reports")
        outputs_folder = os.path.join(os.path.dirname(__file__), "outputs")
        clear_folder(reports_folder)
        clear_folder(outputs_folder)
        
        pdf_filename = os.path.join(reports_folder, f"report_{int(time.time())}.pdf")
        with open(pdf_filename, "wb") as f:
            f.write(uploaded_pdf.getbuffer())
        st.success(f"Document saved to {pdf_filename}")
        
        st.markdown("#### Processing Document...")
        run_pipeline()
        
        # Verify that docexplore.json was created
        docexplore_path = os.path.join(outputs_folder, "docexplore.json")
        if not os.path.exists(docexplore_path):
            st.error(f"Pipeline failed to generate {docexplore_path}. Check the pipeline scripts for errors.")
            st.stop()
        
        st.session_state.doc_uploaded = True
        st.rerun()
else:
    # Analysis Page
    try:
        docexplore_data = load_json("outputs/docexplore.json")
    except Exception as e:
        st.error(f"Error loading analysis outputs: {e}")
        st.stop()
    
    clusters = docexplore_data.get("clusters", [])
    if not clusters:
        st.error("No clusters found in the document. Please ensure the backend pipeline is working correctly.")
        st.stop()
    
    topics_dict = {cluster["id"]: cluster["name"] for cluster in clusters}
    clustered_data = []
    for cluster in clusters:
        for item in cluster["items"]:
            clustered_data.append(item)
    
    cluster_ids = [item.get("cluster") for item in clustered_data if "cluster" in item]
    counts = Counter(cluster_ids)
    
    data_rows = []
    for cid, topic_name in topics_dict.items():
        mention_count = counts.get(cid, 0)
        data_rows.append({
            "TopicName": topic_name,
            "Mentions": mention_count,
            "ClusterID": cid
        })
    
    df_topics = pd.DataFrame(data_rows)
    if df_topics.empty or "Mentions" not in df_topics.columns:
        st.error("No topics extracted. Check the backend pipeline.")
        st.stop()
    
    df_topics.sort_values("Mentions", ascending=False, inplace=True)
    
    total_topics = len(df_topics)
    top_topic_name = df_topics.iloc[0]["TopicName"] if not df_topics.empty else "N/A"
    top_topic_mentions = df_topics.iloc[0]["Mentions"] if not df_topics.empty else 0
    
    topic_keywords = extract_keywords_per_topic(clustered_data, topics_dict)
    topic_overlaps = compute_topic_overlap(clustered_data, topics_dict)
    all_keywords = extract_all_keywords(clustered_data)
    
    doc_title = docexplore_data.get("title", "Document Insights")
    doc_desc = docexplore_data.get("description", "")
    st.title(doc_title)
    if doc_desc:
        st.markdown(f"**Description:** {doc_desc}")
    
    # Key Insights
    st.markdown("### Key Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div style="background-color:#1E1E1E; padding:1.5rem; border-radius:0.5rem; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
                <h5 style="color:#2E91E5; margin:0;">Total Topics</h5>
                <p style="font-size:1.5rem; font-weight:bold; color:#f0f0f0; margin:0.5rem 0;">{total_topics}</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col2:
        st.markdown(
            f"""
            <div style="background-color:#1E1E1E; padding:1.5rem; border-radius:0.5rem; text-align:center; box-shadow: 0 4px 6px rgba(0,0,0,0.3); min-height:150px; display:flex; flex-direction:column; justify-content:center;">
                <h5 style="color:#2E91E5; margin:0;">Most Discussed</h5>
                <p style="font-size:0.9rem; font-weight:500; color:#f0f0f0; margin:0.5rem 0; line-height:1.2;">{top_topic_name}</p>
                <p style="font-size:0.9rem; color:#888;">{top_topic_mentions} mentions</p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Tabs
    tabs = st.tabs(["Overview", "Mentions", "Detailed Analysis", "Visualizations"])
    
    with tabs[0]:
        st.subheader("Overview")
        st.markdown("Explore the key topics extracted from the document with their mention counts.")
        if df_topics.empty:
            st.write("No topics found.")
        else:
            max_m = df_topics["Mentions"].max() or 1
            for _, row in df_topics.iterrows():
                tname = row["TopicName"]
                m_count = row["Mentions"]
                bar_pct = int((m_count / max_m) * 100)
                st.markdown(
                    f"""
                    <div style="background-color:#1E1E1E; padding:1rem; border-radius:0.5rem; margin-bottom:1.2rem; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                        <div style="display:flex; justify-content:space-between; font-weight:500; color:#f0f0f0;">
                            <span>{tname}</span>
                            <span style="color:#2E91E5;">{m_count} mentions</span>
                        </div>
                        <div style="background-color:#333; border-radius:5px; height:10px; width:100%; margin-top:8px;">
                            <div style="background-color:#2E91E5; width:{bar_pct}%; height:100%; border-radius:5px; transition: width 0.3s ease-in-out;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    with tabs[1]:
        st.subheader("Mentions")
        st.markdown("Select a topic to view its document excerpts.")
        if df_topics.empty:
            st.write("No topics available.")
        else:
            topic_options = df_topics["TopicName"].tolist()
            selected_topic = st.selectbox("Choose a Topic", topic_options, key="mentions_select")
            row_match = df_topics[df_topics["TopicName"] == selected_topic].iloc[0]
            sel_cid = row_match["ClusterID"]
            st.markdown(f"### Excerpts for: **{selected_topic}**")
            filtered_items = [item for item in clustered_data if item.get("cluster") == sel_cid]
            if filtered_items:
                for item in filtered_items:
                    st.markdown(f"**Document ID:** {item.get('id')}")
                    excerpt = item.get("text", "").replace("\n", " ")[:500] + "..."
                    st.markdown(f"**Excerpt:** {excerpt}")
                    st.markdown("---")
            else:
                st.write("No excerpts available for this topic.")
    
    with tabs[2]:
        st.subheader("Detailed Analysis")
        st.markdown("Dive deeper into topic insights, including top keywords and overlaps with other topics.")
        if df_topics.empty:
            st.write("No topics available for analysis.")
        else:
            topic_options = df_topics["TopicName"].tolist()
            selected_topic = st.selectbox("Choose a Topic for Analysis", topic_options, key="analysis_select")
            
            st.markdown(f"#### Top Keywords for **{selected_topic}**")
            keywords = topic_keywords.get(selected_topic, [])
            if keywords:
                st.markdown(
                    f"""
                    <div style="display:flex; flex-wrap:wrap; gap:0.5rem;">
                        {''.join([f'<span style="background-color:#2E91E5; color:#fff; padding:0.3rem 0.8rem; border-radius:15px; font-size:0.9rem;">{kw}</span>' for kw in keywords])}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.write("No keywords extracted.")
            
            st.markdown("#### Topic Overlaps")
            relevant_overlaps = [(t1, t2, overlap) for t1, t2, overlap in topic_overlaps if t1 == selected_topic or t2 == selected_topic]
            if relevant_overlaps:
                for t1, t2, overlap in relevant_overlaps:
                    other_topic = t2 if t1 == selected_topic else t1
                    overlap_pct = overlap * 100
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; margin-bottom:0.8rem;">
                            <span style="flex:1;">{other_topic}</span>
                            <div style="background-color:#333; border-radius:5px; height:8px; width:100px; margin:0 1rem;">
                                <div style="background-color:#2E91E5; width:{overlap_pct}%; height:100%; border-radius:5px;"></div>
                            </div>
                            <span style="color:#2E91E5;">{overlap_pct:.1f}%</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.write("No significant overlaps found with other topics.")
    
    with tabs[3]:
        st.subheader("Visualizations")
        st.markdown("Explore interactive visualizations to gain deeper insights into the document's topics and keywords.")
        
        st.markdown("#### Topic Explorer")
        positions, topic_names = compute_topic_positions(clustered_data, topics_dict)
        if positions is not None and topic_names:
            bubble_data = []
            for i, (x, y) in enumerate(positions):
                topic_name = topic_names[i]
                mentions = df_topics[df_topics["TopicName"] == topic_name]["Mentions"].iloc[0]
                bubble_data.append({
                    "TopicName": topic_name,
                    "x": x,
                    "y": y,
                    "Mentions": mentions
                })
            bubble_df = pd.DataFrame(bubble_data)
            
            fig_bubble = px.scatter(
                bubble_df,
                x="x",
                y="y",
                size="Mentions",
                color="Mentions",
                color_continuous_scale="Blues",
                hover_data={"TopicName": True, "Mentions": True},
                text="TopicName",
                size_max=60
            )
            fig_bubble.update_traces(
                textposition="middle center",
                textfont=dict(size=10, color="#fff"),
                marker=dict(line=dict(width=1, color="#fff"))
            )
            fig_bubble.update_layout(
                title="Topic Explorer: Size by Mentions, Position by Similarity",
                plot_bgcolor="#1E1E1E",
                paper_bgcolor="#1E1E1E",
                font=dict(color="#f0f0f0"),
                title_x=0.5,
                showlegend=False,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                height=600
            )
            st.plotly_chart(fig_bubble, use_container_width=True)
        else:
            st.error("Unable to generate Topic Explorer. Ensure embeddings are included in 'outputs/docexplore.json' from the backend pipeline.")
        
        st.markdown("#### Top Keywords Across All Topics")
        if all_keywords:
            word_cloud_html = "<div style='display:flex; flex-wrap:wrap; gap:15px; justify-content:center; padding:1rem;'>"
            max_freq = max(freq for _, freq in all_keywords)
            for word, freq in all_keywords:
                font_size = 14 + (freq / max_freq) * 32  # Scale font size between 14 and 46
                word_cloud_html += f'<span style="font-size:{font_size}px; color:#2E91E5; padding:5px 10px; display:inline-block;">{word}</span>'
            word_cloud_html += "</div>"
            st.markdown(word_cloud_html, unsafe_allow_html=True)
        else:
            st.write("No keywords available for visualization.")
    
    st.markdown("---")
    st.caption("Click the button below to close the current document session and start over.")
    if st.button("Close Document Session"):
        st.session_state.clear()
        st.rerun()