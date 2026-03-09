"""Streamlit frontend for RAG Knowledge Assistant"""

import streamlit as st
import requests
import uuid
from datetime import datetime
from typing import Optional

# Configuration
API_BASE_URL = "http://localhost:8000"
BACKEND_TIMEOUT = 30

# Page config
st.set_page_config(
    page_title="RAG Knowledge Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .message-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .assistant-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
    .citation {
        background-color: #fff3e0;
        padding: 0.5rem;
        border-left: 3px solid #FF9800;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

# Header
st.title("🤖 RAG Knowledge Assistant")
st.markdown("**Your AI-powered document assistant with source citations**")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    
    top_k = st.slider(
        "Context Chunks (top-k)",
        min_value=1,
        max_value=10,
        value=5,
        help="Number of document chunks to use for context"
    )
    
    if st.button("🔄 New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.chat_messages = []
        st.rerun()
    
    if st.button("📋 Clear Chats", use_container_width=True):
        st.session_state.chat_messages = []
        st.rerun()
    
    st.divider()
    st.markdown("**Session Info:**")
    st.code(st.session_state.session_id[:12] + "...")
    st.markdown(f"**Messages**: {len(st.session_state.chat_messages)}")

# Main content
tabs = st.tabs(["📂 Documents", "💬 Chat"])

# TAB 1: Documents
with tabs[0]:
    st.header("Document Management")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        uploaded_files = st.file_uploader(
            "Upload documents (PDF, DOCX, TXT, MD)",
            type=["pdf", "docx", "txt", "md"],
            accept_multiple_files=True,
            help="Upload documents to build your knowledge base"
        )
    
    with col2:
        if st.button("📤 Upload", use_container_width=True):
            if uploaded_files:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, file in enumerate(uploaded_files):
                    status_text.text(f"Uploading {file.name}...")
                    
                    try:
                        # Upload file
                        files = {"file": (file.name, file, file.type)}
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/documents/upload",
                            files=files,
                            timeout=BACKEND_TIMEOUT
                        )
                        
                        if response.status_code == 200:
                            doc_data = response.json()
                            st.session_state.documents.append(doc_data)
                            status_text.success(f"✅ {file.name} uploaded successfully!")
                        else:
                            st.error(f"❌ Failed to upload {file.name}: {response.text}")
                    
                    except Exception as e:
                        st.error(f"❌ Error uploading {file.name}: {str(e)}")
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                progress_bar.empty()
                status_text.empty()
    
    st.divider()
    
    # Load existing documents
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/documents",
            timeout=BACKEND_TIMEOUT
        )
        if response.status_code == 200:
            documents = response.json()
            if documents:
                st.success(f"**{len(documents)} documents in knowledge base**")
                
                # Display documents
                for doc in documents:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"📄 **{doc['filename']}**")
                        st.caption(f"Size: {doc['size_bytes']:,} bytes | Chunks: {doc['chunks_count']}")
                    
                    with col2:
                        st.caption(f"📅 {doc['uploaded_at']}")
                    
                    with col3:
                        if st.button("🗑️", key=f"delete_{doc['doc_id']}", use_container_width=True):
                            try:
                                del_response = requests.delete(
                                    f"{API_BASE_URL}/api/v1/documents/{doc['doc_id']}",
                                    timeout=BACKEND_TIMEOUT
                                )
                                if del_response.status_code == 200:
                                    st.success("Document deleted")
                                    st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting: {str(e)}")
            else:
                st.info("📭 No documents yet. Upload some files to get started!")
    
    except Exception as e:
        st.error(f"Error loading documents: {str(e)}")

# TAB 2: Chat
with tabs[1]:
    st.header("Chat with Your Documents")
    
    # Chat history display
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_messages:
            if message["role"] == "user":
                st.markdown(
                    f'<div class="message-box user-message"><b>You:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="message-box assistant-message"><b>Assistant:</b> {message["content"]}</div>',
                    unsafe_allow_html=True
                )
                
                # Display citations
                if "citations" in message and message["citations"]:
                    with st.expander("📚 Sources"):
                        for citation in message["citations"]:
                            source_text = f"**{citation['source']}**"
                            if citation.get('page'):
                                source_text += f" (Page {citation['page']})"
                            
                            st.markdown(
                                f'<div class="citation">{source_text}<br/><em>{citation.get("content_snippet", "")[:100]}...</em></div>',
                                unsafe_allow_html=True
                            )
    
    st.divider()
    
    # Chat input
    col1, col2 = st.columns([20, 1])
    
    with col1:
        user_input = st.text_input(
            "Ask a question about your documents...",
            placeholder="E.g., What is the company's return policy?",
            label_visibility="collapsed"
        )
    
    with col2:
        send_button = st.button("Send", use_container_width=True)
    
    # Process input
    if send_button and user_input:
        try:
            # Add user message to history
            st.session_state.chat_messages.append({
                "role": "user",
                "content": user_input
            })
            
            # Get response from API
            with st.spinner("🔍 Thinking..."):
                response = requests.post(
                    f"{API_BASE_URL}/api/v1/chat/query",
                    json={
                        "query": user_input,
                        "session_id": st.session_state.session_id,
                        "top_k": top_k
                    },
                    timeout=BACKEND_TIMEOUT
                )
            
            if response.status_code == 200:
                result = response.json()
                
                # Add assistant message to history
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "citations": result.get("citations", [])
                })
                
                st.rerun()
            else:
                st.error(f"❌ Error: {response.text}")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to backend. Make sure the API is running on http://localhost:8000")
        except requests.exceptions.Timeout:
            st.error(f"❌ Request timed out after {BACKEND_TIMEOUT}s")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    # Empty state
    if not st.session_state.chat_messages:
        st.info("""
        👋 Welcome! 
        
        1. Upload documents in the **Documents** tab
        2. Ask questions here and I'll answer based on your documents
        3. All answers include source citations
        """)
