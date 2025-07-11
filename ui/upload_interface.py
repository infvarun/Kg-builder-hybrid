import streamlit as st
import os
import tempfile
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
import logging

from core.document_processor import DocumentProcessor
from core.graph_manager import GraphManager
from core.embedding_manager import EmbeddingManager
from utils.progress_tracker import ProgressTracker
from utils.cost_calculator import CostCalculator

class UploadInterface:
    """Streamlit interface for document upload and processing."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.document_processor = DocumentProcessor()
        self.graph_manager = GraphManager()
        self.embedding_manager = EmbeddingManager()
        self.progress_tracker = ProgressTracker()
        self.cost_calculator = CostCalculator()

    def render(self):
        """Render the upload interface."""
        st.header("📄 Upload Clinical Documents")

        # File upload section
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=['pdf'],
            help="Upload Clinical IRT Study Design PDF documents"
        )

        if uploaded_file is not None:
            # Display file information
            st.subheader("📋 File Information")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("File Name", uploaded_file.name)
            with col2:
                st.metric("File Size", f"{uploaded_file.size / 1024:.1f} KB")
            with col3:
                st.metric("File Type", uploaded_file.type)

            # Processing options
            st.subheader("⚙️ Processing Options")

            col1, col2 = st.columns(2)
            with col1:
                chunk_size = st.slider("Chunk Size", 500, 2000, 1000, 100)
                chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)
            with col2:
                enable_embeddings = st.checkbox("Generate Embeddings", value=True)
                enable_llm_processing = st.checkbox("LLM Entity Extraction", value=True)

            # Cost estimation
            if st.button("📊 Estimate Processing Cost"):
                with st.spinner("Analyzing document..."):
                    cost_info = self._estimate_processing_cost(uploaded_file)
                    if cost_info:
                        st.subheader("💰 Cost Estimation")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Estimated Chunks", cost_info['total_chunks'])
                        with col2:
                            st.metric("Estimated Tokens", cost_info['estimated_tokens'])
                        with col3:
                            st.metric("Estimated Cost", f"${cost_info['estimated_cost']:.4f}")
                        with col4:
                            st.metric("Cost per Chunk", f"${cost_info['cost_per_chunk']:.4f}")

            # Process document button
            if st.button("🚀 Process Document", type="primary"):
                if self._validate_file(uploaded_file):
                    self._process_document(
                        uploaded_file,
                        chunk_size,
                        chunk_overlap,
                        enable_embeddings,
                        enable_llm_processing
                    )

    def _validate_file(self, uploaded_file) -> bool:
        """Validate uploaded file."""
        if uploaded_file.size > 50 * 1024 * 1024:  # 50MB limit
            st.error("File size exceeds 50MB limit")
            return False

        if not uploaded_file.type == 'application/pdf':
            st.error("Only PDF files are supported")
            return False

        return True

    def _estimate_processing_cost(self, uploaded_file) -> Optional[Dict[str, Any]]:
        """Estimate processing cost for the document."""
        try:
            # Create a temporary file to process
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_file_path = tmp_file.name

            # Reset file pointer
            uploaded_file.seek(0)

            # Extract content
            content = self.document_processor.extract_pdf_content(uploaded_file)

            # Create chunks
            chunks = self.document_processor.create_semantic_chunks(content)

            # Calculate cost
            cost_info = self.document_processor.calculate_processing_cost(chunks)

            # Clean up
            os.unlink(tmp_file_path)

            return cost_info

        except Exception as e:
            st.error(f"Error estimating cost: {str(e)}")
            return None

    def _process_document(self, uploaded_file, chunk_size: int, chunk_overlap: int, 
                         enable_embeddings: bool, enable_llm_processing: bool):
        """Process the uploaded document."""
        try:
            # Initialize progress tracking
            progress_placeholder = st.empty()
            status_placeholder = st.empty()

            # Generate unique document ID
            doc_id = str(uuid.uuid4())

            with st.spinner("Processing document..."):
                # Step 1: Extract content
                status_placeholder.info("📖 Extracting content from PDF...")
                content = self.document_processor.extract_pdf_content(uploaded_file)

                # Step 2: Create chunks
                status_placeholder.info("✂️ Creating semantic chunks...")
                self.document_processor.chunk_size = chunk_size
                self.document_processor.chunk_overlap = chunk_overlap
                chunks = self.document_processor.create_semantic_chunks(content)

                # Step 3: Generate embeddings
                if enable_embeddings:
                    status_placeholder.info("🧠 Generating embeddings...")
                    chunks = self.embedding_manager.batch_process_chunks(chunks)

                # Step 4: Process with LLM (if enabled)
                if enable_llm_processing:
                    status_placeholder.info("🤖 Extracting entities with LLM...")
                    # This would require OpenAI API key
                    st.warning("LLM processing requires OpenAI API key in environment variables")

                # Step 5: Save to Neo4j
                status_placeholder.info("💾 Saving to Neo4j database...")

                # Generate file hash
                file_hash = self.document_processor.generate_document_hash(uploaded_file)

                # Prepare document data
                document_data = {
                    'name': uploaded_file.name,
                    'file_hash': file_hash,
                    'metadata': content['metadata']
                }

                # Save to database
                document_name = self.graph_manager.save_document(document_data, chunks)

                # Step 6: Save upload metadata
                upload_data = {
                    'upload_id': doc_id,
                    'document_name': document_name,
                    'total_chunks': len(chunks),
                    'status': 'completed'
                }

                self.graph_manager.save_upload_metadata(upload_data)

                # Clear status
                status_placeholder.empty()

                # Show success message
                st.success("✅ Document processed successfully!")

                # Display results
                st.subheader("📊 Processing Results")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Pages", content['total_pages'])
                with col2:
                    st.metric("Chunks Created", len(chunks))
                with col3:
                    st.metric("Tables Extracted", len(content['tables']))
                with col4:
                    st.metric("Processing Status", "✅ Completed")

                # Show sample chunks
                if st.checkbox("Show Sample Chunks"):
                    st.subheader("📄 Sample Chunks")
                    for i, chunk in enumerate(chunks[:3]):
                        with st.expander(f"Chunk {i+1} (Page {chunk['page_number']})"):
                            st.text(chunk['content'][:500] + "..." if len(chunk['content']) > 500 else chunk['content'])
                            st.caption(f"Words: {chunk['word_count']} | Characters: {chunk['char_count']} | Type: {chunk['chunk_type']}")

        except Exception as e:
            st.error(f"Error processing document: {str(e)}")
            self.logger.error(f"Document processing error: {str(e)}")