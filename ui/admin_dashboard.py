
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
                            st.session_state[f"show_modal_{doc['name']}"] = True
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
        """Show detailed information about a document in a modal dialog."""
        # Initialize modal state
        if f"show_modal_{document['name']}" not in st.session_state:
            st.session_state[f"show_modal_{document['name']}"] = True
        
        # Modal dialog using st.dialog
        @st.dialog(f"📄 Document Details: {document['name']}", width="large")
        def document_modal():
            # Document header with key metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("📊 Total Chunks", document['total_chunks'])
            with col2:
                st.metric("✅ Actual Chunks", document['actual_chunks'])
            with col3:
                metadata = document.get('metadata', {})
                st.metric("📖 Pages", metadata.get('num_pages', 'N/A'))
            with col4:
                st.metric("📅 Upload Date", document['upload_date'])
            
            st.divider()
            
            # Status and basic info
            col1, col2 = st.columns(2)
            with col1:
                status_color = "🟢" if document['status'] == 'completed' else "🟡"
                st.markdown(f"**Status:** {status_color} {document['status'].title()}")
                
                if metadata.get('title'):
                    st.markdown(f"**Title:** {metadata['title']}")
                
                if metadata.get('author'):
                    st.markdown(f"**Author:** {metadata['author']}")
            
            with col2:
                if metadata.get('creation_date'):
                    st.markdown(f"**Created:** {metadata['creation_date']}")
                
                if metadata.get('file_size'):
                    file_size_mb = metadata['file_size'] / (1024 * 1024)
                    st.markdown(f"**File Size:** {file_size_mb:.1f} MB")
                
                if metadata.get('format'):
                    st.markdown(f"**Format:** {metadata['format']}")
            
            # Full metadata in expandable section
            if document.get('metadata'):
                with st.expander("🔍 Complete Metadata", expanded=False):
                    metadata_items = []
                    for k, v in document['metadata'].items():
                        metadata_items.append({
                            'Property': k.replace('_', ' ').title(), 
                            'Value': str(v)
                        })
                    
                    if metadata_items:
                        metadata_df = pd.DataFrame(metadata_items)
                        st.dataframe(
                            metadata_df, 
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Property": st.column_config.TextColumn("Property", width="medium"),
                                "Value": st.column_config.TextColumn("Value", width="large")
                            }
                        )
            
            # Processing information
            st.subheader("⚙️ Processing Information")
            
            processing_col1, processing_col2 = st.columns(2)
            with processing_col1:
                chunk_efficiency = (document['actual_chunks'] / document['total_chunks'] * 100) if document['total_chunks'] > 0 else 0
                st.metric("📈 Chunk Efficiency", f"{chunk_efficiency:.1f}%")
                
                if metadata.get('processing_time'):
                    st.metric("⏱️ Processing Time", f"{metadata['processing_time']:.1f}s")
            
            with processing_col2:
                if metadata.get('embeddings_generated'):
                    st.metric("🧠 Embeddings", "✅ Generated" if metadata['embeddings_generated'] else "❌ Not Generated")
                
                if metadata.get('entities_extracted'):
                    st.metric("🏷️ Entities", "✅ Extracted" if metadata['entities_extracted'] else "❌ Not Extracted")
            
            # Chunk preview
            st.subheader("📄 Chunk Preview")
            try:
                # Get sample chunks for this document
                sample_chunks = self.graph_manager.get_document_chunks(document['name'], limit=3)
                
                if sample_chunks:
                    for i, chunk in enumerate(sample_chunks):
                        with st.expander(f"Sample Chunk {i+1} - Page {chunk.get('page_number', 'N/A')}", expanded=i == 0):
                            chunk_col1, chunk_col2, chunk_col3 = st.columns(3)
                            with chunk_col1:
                                st.caption(f"Type: {chunk.get('chunk_type', 'N/A')}")
                            with chunk_col2:
                                st.caption(f"Words: {chunk.get('word_count', 'N/A')}")
                            with chunk_col3:
                                st.caption(f"Characters: {chunk.get('char_count', 'N/A')}")
                            
                            content = chunk.get('content', '')
                            preview = content[:500] + "..." if len(content) > 500 else content
                            st.text_area(
                                "Content Preview",
                                preview,
                                height=120,
                                key=f"chunk_preview_{i}",
                                disabled=True
                            )
                else:
                    st.info("No chunk data available for preview.")
            except Exception as e:
                st.warning(f"Could not load chunk preview: {str(e)}")
            
            # Action buttons
            st.divider()
            action_col1, action_col2, action_col3 = st.columns(3)
            
            with action_col1:
                if st.button("🔄 Reprocess Document", type="secondary", use_container_width=True):
                    st.info("Document reprocessing would be implemented here.")
            
            with action_col2:
                if st.button("📊 View Analytics", type="secondary", use_container_width=True):
                    st.info("Document analytics would be displayed here.")
            
            with action_col3:
                if st.button("❌ Delete Document", type="secondary", use_container_width=True):
                    if st.checkbox("Confirm deletion", key="delete_confirm_modal"):
                        try:
                            success = self.graph_manager.delete_document(document['name'])
                            if success:
                                st.success("Document deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Failed to delete document")
                        except Exception as e:
                            st.error(f"Error deleting document: {str(e)}")
        
        # Show the modal
        if st.session_state[f"show_modal_{document['name']}"]:
            document_modal()
    
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
