
import os
import logging
from typing import List, Dict, Any, Optional
import pdfplumber
import PyPDF2
from io import BytesIO
import re
from datetime import datetime
import hashlib

class DocumentProcessor:
    """Handles PDF document processing, chunking, and metadata extraction."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
    def extract_pdf_content(self, pdf_file: BytesIO) -> Dict[str, Any]:
        """Extract content and metadata from PDF file."""
        try:
            # Reset file pointer
            pdf_file.seek(0)
            
            # Extract metadata using PyPDF2
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            metadata = {
                'title': pdf_reader.metadata.get('/Title', '') if pdf_reader.metadata else '',
                'author': pdf_reader.metadata.get('/Author', '') if pdf_reader.metadata else '',
                'creator': pdf_reader.metadata.get('/Creator', '') if pdf_reader.metadata else '',
                'producer': pdf_reader.metadata.get('/Producer', '') if pdf_reader.metadata else '',
                'creation_date': pdf_reader.metadata.get('/CreationDate', '') if pdf_reader.metadata else '',
                'modification_date': pdf_reader.metadata.get('/ModDate', '') if pdf_reader.metadata else '',
                'num_pages': len(pdf_reader.pages)
            }
            
            # Reset file pointer for pdfplumber
            pdf_file.seek(0)
            
            # Extract text and structure using pdfplumber
            pages_content = []
            tables = []
            
            with pdfplumber.open(pdf_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_content = {
                        'page_number': page_num + 1,
                        'text': page.extract_text() or '',
                        'tables': [],
                        'paragraphs': []
                    }
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            if table:
                                table_data = {
                                    'table_id': f"table_{page_num + 1}_{table_idx + 1}",
                                    'page_number': page_num + 1,
                                    'data': table
                                }
                                page_content['tables'].append(table_data)
                                tables.append(table_data)
                    
                    # Extract paragraphs
                    if page_content['text']:
                        paragraphs = self._extract_paragraphs(page_content['text'])
                        page_content['paragraphs'] = paragraphs
                    
                    pages_content.append(page_content)
            
            return {
                'metadata': metadata,
                'pages': pages_content,
                'tables': tables,
                'total_pages': len(pages_content)
            }
            
        except Exception as e:
            self.logger.error(f"Error extracting PDF content: {str(e)}")
            raise
    
    def _extract_paragraphs(self, text: str) -> List[Dict[str, Any]]:
        """Extract paragraphs from text while preserving structure."""
        paragraphs = []
        
        # Split by double newlines to identify paragraphs
        raw_paragraphs = re.split(r'\n\s*\n', text)
        
        for idx, paragraph in enumerate(raw_paragraphs):
            if paragraph.strip():
                # Detect bullet points
                is_bullet = bool(re.match(r'^\s*[-•*]\s+', paragraph.strip()))
                
                # Detect numbered lists
                is_numbered = bool(re.match(r'^\s*\d+\.\s+', paragraph.strip()))
                
                paragraphs.append({
                    'paragraph_id': idx + 1,
                    'text': paragraph.strip(),
                    'is_bullet': is_bullet,
                    'is_numbered': is_numbered,
                    'word_count': len(paragraph.split())
                })
        
        return paragraphs
    
    def create_semantic_chunks(self, document_content: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic chunks from document content."""
        chunks = []
        chunk_id = 1
        
        for page in document_content['pages']:
            page_num = page['page_number']
            
            # Process paragraphs
            current_chunk = ""
            current_chunk_paras = []
            
            for paragraph in page['paragraphs']:
                para_text = paragraph['text']
                
                # Check if adding this paragraph would exceed chunk size
                if len(current_chunk + para_text) > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunks.append({
                        'chunk_id': chunk_id,
                        'content': current_chunk.strip(),
                        'page_number': page_num,
                        'paragraph_numbers': current_chunk_paras,
                        'word_count': len(current_chunk.split()),
                        'char_count': len(current_chunk),
                        'chunk_type': 'paragraph'
                    })
                    
                    chunk_id += 1
                    
                    # Start new chunk with overlap
                    if self.chunk_overlap > 0:
                        overlap_words = current_chunk.split()[-self.chunk_overlap:]
                        current_chunk = ' '.join(overlap_words) + ' ' + para_text
                    else:
                        current_chunk = para_text
                    
                    current_chunk_paras = [paragraph['paragraph_id']]
                else:
                    current_chunk += ' ' + para_text if current_chunk else para_text
                    current_chunk_paras.append(paragraph['paragraph_id'])
            
            # Add remaining chunk
            if current_chunk.strip():
                chunks.append({
                    'chunk_id': chunk_id,
                    'content': current_chunk.strip(),
                    'page_number': page_num,
                    'paragraph_numbers': current_chunk_paras,
                    'word_count': len(current_chunk.split()),
                    'char_count': len(current_chunk),
                    'chunk_type': 'paragraph'
                })
                chunk_id += 1
            
            # Process tables separately
            for table in page['tables']:
                table_content = self._table_to_text(table['data'])
                chunks.append({
                    'chunk_id': chunk_id,
                    'content': table_content,
                    'page_number': page_num,
                    'paragraph_numbers': [],
                    'word_count': len(table_content.split()),
                    'char_count': len(table_content),
                    'chunk_type': 'table',
                    'table_id': table['table_id']
                })
                chunk_id += 1
        
        return chunks
    
    def _table_to_text(self, table_data: List[List[str]]) -> str:
        """Convert table data to text format."""
        if not table_data:
            return ""
        
        text_lines = []
        for row in table_data:
            if row:
                row_text = ' | '.join([str(cell) if cell else '' for cell in row])
                text_lines.append(row_text)
        
        return '\n'.join(text_lines)
    
    def calculate_processing_cost(self, chunks: List[Dict[str, Any]], cost_per_token: float = 0.0001) -> Dict[str, Any]:
        """Calculate estimated processing cost for chunks."""
        total_tokens = sum(chunk['word_count'] * 1.3 for chunk in chunks)  # Approximate token count
        total_cost = total_tokens * cost_per_token
        
        return {
            'total_chunks': len(chunks),
            'estimated_tokens': int(total_tokens),
            'estimated_cost': round(total_cost, 4),
            'cost_per_chunk': round(total_cost / len(chunks), 4) if chunks else 0
        }
    
    def generate_document_hash(self, pdf_file: BytesIO) -> str:
        """Generate unique hash for document."""
        pdf_file.seek(0)
        content = pdf_file.read()
        pdf_file.seek(0)
        return hashlib.md5(content).hexdigest()
