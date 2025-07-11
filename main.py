
import streamlit as st
import sys
import os
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from ui.upload_interface import UploadInterface
from ui.admin_dashboard import AdminDashboard
from ui.search_interface import SearchInterface
from core.graph_manager import GraphManager
from config.neo4j_config import Neo4jConfig

def main():
    st.set_page_config(
        page_title="Clinical IRT Study Design Converter",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🏥 Clinical IRT Study Design to Neo4j Graph Converter")
    st.markdown("Convert Clinical IRT Study Design PDFs into Neo4j knowledge graphs with intelligent chunking and semantic search.")
    
    # Initialize Neo4j connection
    try:
        graph_manager = GraphManager()
        graph_manager.initialize_database()
    except Exception as e:
        st.error(f"Failed to connect to Neo4j: {str(e)}")
        st.stop()
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    
    # Initialize session state for page if not exists
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Upload Documents"
    
    # Navigation buttons
    if st.sidebar.button("📄 Upload Documents", use_container_width=True):
        st.session_state.current_page = "Upload Documents"
    
    if st.sidebar.button("🏗️ Admin Dashboard", use_container_width=True):
        st.session_state.current_page = "Admin Dashboard"
    
    if st.sidebar.button("🔍 Search Documents", use_container_width=True):
        st.session_state.current_page = "Search Documents"
    
    # Page routing
    if st.session_state.current_page == "Upload Documents":
        upload_interface = UploadInterface()
        upload_interface.render()
    elif st.session_state.current_page == "Admin Dashboard":
        admin_dashboard = AdminDashboard()
        admin_dashboard.render()
    elif st.session_state.current_page == "Search Documents":
        search_interface = SearchInterface()
        search_interface.render()

if __name__ == "__main__":
    main()
