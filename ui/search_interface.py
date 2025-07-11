
import streamlit as st
from typing import List, Dict, Any, Optional
import logging

from core.graph_manager import GraphManager
from core.embedding_manager import EmbeddingManager

class SearchInterface:
    """Search interface for document content."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.graph_manager = GraphManager()
        self.embedding_manager = EmbeddingManager()
    
    def render(self):
        """Render the search interface."""
        st.header("🔍 Search Documents")
        
        # Search input
        search_query = st.text_input(
            "Enter your search query",
            placeholder="Search for clinical terms, procedures, medications, etc."
        )
        
        # Search options
        col1, col2, col3 = st.columns(3)
        with col1:
            search_type = st.selectbox(
                "Search Type",
                ["Text Search", "Semantic Search", "Combined"]
            )
        with col2:
            max_results = st.number_input(
                "Max Results",
                min_value=1,
                max_value=50,
                value=10
            )
        with col3:
            include_context = st.checkbox("Include Context", value=True)
        
        # Search button
        if st.button("🔍 Search", type="primary") and search_query:
            self._perform_search(search_query, search_type, max_results, include_context)
        
        # Advanced search options
        with st.expander("🔧 Advanced Search Options"):
            st.subheader("Filters")
            
            # Document filter
            documents = self.graph_manager.get_all_documents()
            doc_names = [doc['name'] for doc in documents]
            
            selected_docs = st.multiselect(
                "Filter by Documents",
                doc_names,
                default=doc_names
            )
            
            # Page range filter
            col1, col2 = st.columns(2)
            with col1:
                min_page = st.number_input("Min Page", min_value=1, value=1)
            with col2:
                max_page = st.number_input("Max Page", min_value=1, value=999)
            
            # Chunk type filter
            chunk_types = st.multiselect(
                "Chunk Types",
                ["paragraph", "table", "all"],
                default=["paragraph", "table"]
            )
        
        # Search history
        if 'search_history' not in st.session_state:
            st.session_state.search_history = []
        
        if st.session_state.search_history:
            st.subheader("📚 Recent Searches")
            for i, query in enumerate(st.session_state.search_history[-5:]):
                if st.button(f"🔄 {query}", key=f"history_{i}"):
                    st.session_state.search_query = query
                    st.rerun()
    
    def _perform_search(self, query: str, search_type: str, max_results: int, include_context: bool):
        """Perform search based on query and type."""
        try:
            with st.spinner("Searching..."):
                if search_type == "Text Search":
                    results = self._text_search(query, max_results)
                elif search_type == "Semantic Search":
                    results = self._semantic_search(query, max_results)
                else:  # Combined
                    text_results = self._text_search(query, max_results // 2)
                    semantic_results = self._semantic_search(query, max_results // 2)
                    results = self._combine_results(text_results, semantic_results)
                
                # Add to search history
                if query not in st.session_state.search_history:
                    st.session_state.search_history.append(query)
                
                # Display results
                self._display_results(results, include_context)
        
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            self.logger.error(f"Search error: {str(e)}")
    
    def _text_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform text-based search."""
        return self.graph_manager.search_chunks(query, max_results)
    
    def _semantic_search(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Perform semantic search using embeddings."""
        # This would require getting all chunks with embeddings
        # For now, return text search results
        return self.graph_manager.search_chunks(query, max_results)
    
    def _combine_results(self, text_results: List[Dict[str, Any]], 
                        semantic_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine text and semantic search results."""
        combined = []
        seen_chunks = set()
        
        # Add text results first
        for result in text_results:
            chunk_id = result['chunk_id']
            if chunk_id not in seen_chunks:
                result['search_type'] = 'text'
                combined.append(result)
                seen_chunks.add(chunk_id)
        
        # Add semantic results
        for result in semantic_results:
            chunk_id = result['chunk_id']
            if chunk_id not in seen_chunks:
                result['search_type'] = 'semantic'
                combined.append(result)
                seen_chunks.add(chunk_id)
        
        return combined
    
    def _display_results(self, results: List[Dict[str, Any]], include_context: bool):
        """Display search results."""
        if not results:
            st.info("No results found for your query.")
            return
        
        st.subheader(f"🎯 Search Results ({len(results)} found)")
        
        for i, result in enumerate(results):
            with st.expander(f"Result {i+1} - {result['document_name']} (Page {result['page_number']})"):
                # Result metadata
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Page", result['page_number'])
                with col2:
                    st.metric("Chunk Type", result['chunk_type'])
                with col3:
                    if 'similarity_score' in result:
                        st.metric("Similarity", f"{result['similarity_score']:.3f}")
                
                # Content
                st.subheader("📄 Content")
                content = result['content']
                
                # Highlight query terms (basic implementation)
                if include_context:
                    # Show full content
                    st.text_area("Content", content, height=150, key=f"content_{i}")
                else:
                    # Show snippet
                    snippet = content[:300] + "..." if len(content) > 300 else content
                    st.text(snippet)
                
                # Citation information
                st.subheader("📚 Citation")
                citation = f"Document: {result['document_name']}, Page: {result['page_number']}, Chunk: {result['chunk_id']}"
                st.code(citation)
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("📋 Copy Citation", key=f"copy_{i}"):
                        st.code(citation)
                with col2:
                    if st.button("🔗 View in Context", key=f"context_{i}"):
                        self._show_chunk_context(result)
    
    def _show_chunk_context(self, chunk: Dict[str, Any]):
        """Show chunk in context of surrounding chunks."""
        st.subheader("📖 Chunk Context")
        
        # This would require querying for surrounding chunks
        st.info("Context view would show surrounding chunks from the same document.")
        
        # Display current chunk
        st.text_area("Current Chunk", chunk['content'], height=200)
    
    def _highlight_text(self, text: str, query: str) -> str:
        """Highlight query terms in text."""
        import re
        
        # Simple highlighting (would need more sophisticated implementation)
        pattern = re.compile(re.escape(query), re.IGNORECASE)
        highlighted = pattern.sub(f"**{query}**", text)
        return highlighted
