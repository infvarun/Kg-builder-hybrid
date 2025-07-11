
import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime
import logging

from core.graph_manager import GraphManager

class AdminDashboard:
    """Admin dashboard for managing documents and viewing statistics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph_manager = GraphManager()
    
    def render(self):
        """Render the admin dashboard."""
        st.header("🏗️ Admin Dashboard")
        
        # Get statistics
        stats = self.graph_manager.get_statistics()
        
        # Top statistics cards
        st.subheader("📊 Database Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Documents", stats['total_documents'])
        with col2:
            st.metric("Total Chunks", stats['total_chunks'])
        with col3:
            st.metric("Total Words", f"{stats['total_words']:,}")
        with col4:
            st.metric("Total Triples", stats['total_triples'])
        
        # Document management section
        st.subheader("📂 Document Management")
        
        # Get all documents
        documents = self.graph_manager.get_all_documents()
        
        if not documents:
            st.info("No documents found in the database.")
            return
        
        # Display documents as cards
        for doc in documents:
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**{doc['name']}**")
                    st.caption(f"Uploaded: {doc['upload_date'][:10]}")
                    st.caption(f"Status: {doc['status']}")
                
                with col2:
                    st.metric("Chunks", doc['total_chunks'])
                
                with col3:
                    # File size from metadata
                    metadata = doc.get('metadata', {})
                    pages = metadata.get('num_pages', 'N/A')
                    st.metric("Pages", pages)
                
                with col4:
                    # Action buttons
                    col4a, col4b = st.columns(2)
                    with col4a:
                        if st.button("🔍", key=f"view_{doc['name']}", help="View Details"):
                            self._show_document_details(doc)
                    with col4b:
                        if st.button("🗑️", key=f"delete_{doc['name']}", help="Delete Document"):
                            self._delete_document(doc['name'])
                
                st.divider()
        
        # Bulk operations
        st.subheader("🔧 Bulk Operations")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔄 Refresh Statistics"):
                st.rerun()
        
        with col2:
            if st.button("🧹 Cleanup Database", type="secondary"):
                self._cleanup_database()
    
    def _show_document_details(self, document: Dict[str, Any]):
        """Show detailed information about a document."""
        st.subheader(f"📄 Document Details: {document['name']}")
        
        # Basic information
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Upload Date:**", document['upload_date'])
            st.write("**Status:**", document['status'])
            st.write("**Total Chunks:**", document['total_chunks'])
        
        with col2:
            st.write("**Actual Chunks:**", document['actual_chunks'])
            metadata = document.get('metadata', {})
            st.write("**Pages:**", metadata.get('num_pages', 'N/A'))
            st.write("**Title:**", metadata.get('title', 'N/A'))
        
        # Metadata
        if document.get('metadata'):
            st.subheader("📋 Document Metadata")
            metadata_df = pd.DataFrame([
                {'Property': k, 'Value': str(v)} 
                for k, v in document['metadata'].items()
            ])
            st.dataframe(metadata_df, use_container_width=True)
        
        # Chunks information
        st.subheader("📄 Chunks Information")
        # This would require additional query to get chunk details
        st.info("Chunk details would be displayed here with additional database queries.")
    
    def _delete_document(self, document_name: str):
        """Delete a document with confirmation."""
        if st.checkbox(f"Confirm deletion of '{document_name}'"):
            if st.button("❌ Delete Permanently", type="secondary"):
                try:
                    success = self.graph_manager.delete_document(document_name)
                    if success:
                        st.success(f"Document '{document_name}' deleted successfully!")
                        st.rerun()
                    else:
                        st.error(f"Failed to delete document '{document_name}'")
                except Exception as e:
                    st.error(f"Error deleting document: {str(e)}")
    
    def _cleanup_database(self):
        """Cleanup database operations."""
        st.subheader("🧹 Database Cleanup")
        
        # Option to remove orphaned nodes
        if st.checkbox("Remove orphaned chunks"):
            if st.button("Execute Cleanup"):
                try:
                    # This would require additional cleanup queries
                    st.info("Cleanup operations would be implemented here")
                except Exception as e:
                    st.error(f"Cleanup error: {str(e)}")
        
        # Option to rebuild indexes
        if st.checkbox("Rebuild database indexes"):
            if st.button("Rebuild Indexes"):
                try:
                    self.graph_manager.initialize_database()
                    st.success("Database indexes rebuilt successfully!")
                except Exception as e:
                    st.error(f"Error rebuilding indexes: {str(e)}")
    
    def _export_statistics(self):
        """Export statistics to CSV."""
        stats = self.graph_manager.get_statistics()
        documents = self.graph_manager.get_all_documents()
        
        # Create summary dataframe
        summary_data = {
            'Metric': ['Total Documents', 'Total Chunks', 'Total Words', 'Total Triples'],
            'Value': [stats['total_documents'], stats['total_chunks'], 
                     stats['total_words'], stats['total_triples']]
        }
        
        summary_df = pd.DataFrame(summary_data)
        
        # Document details dataframe
        doc_data = []
        for doc in documents:
            doc_data.append({
                'Document Name': doc['name'],
                'Upload Date': doc['upload_date'],
                'Status': doc['status'],
                'Total Chunks': doc['total_chunks'],
                'Actual Chunks': doc['actual_chunks']
            })
        
        docs_df = pd.DataFrame(doc_data)
        
        # Download buttons
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Download Summary",
                summary_df.to_csv(index=False),
                file_name="database_summary.csv",
                mime="text/csv"
            )
        with col2:
            st.download_button(
                "Download Documents",
                docs_df.to_csv(index=False),
                file_name="documents_list.csv",
                mime="text/csv"
            )
