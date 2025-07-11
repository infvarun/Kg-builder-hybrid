
import os
import logging
from typing import List, Dict, Any, Optional
import openai
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import json
import re

class LLMProcessor:
    """Handles LLM operations for clinical document processing."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI API
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        self.llm = OpenAI(
            openai_api_key=api_key,
            temperature=0.1,
            max_tokens=2000
        )
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=api_key
        )
        
        # Clinical-specific extraction prompts
        self.clinical_prompt = """
        You are a clinical research expert. Extract key medical and clinical information from the following text.
        
        Focus on:
        1. Medical entities (procedures, medications, conditions, medical devices)
        2. Study phases, timelines, and protocols
        3. Regulatory requirements and compliance points
        4. Investigator roles and responsibilities
        5. Patient populations and inclusion/exclusion criteria
        6. Primary and secondary endpoints
        7. Data collection methods and schedules
        
        Return the information as structured JSON with the following format:
        {
            "medical_entities": [
                {"entity": "entity_name", "type": "procedure|medication|condition|device", "context": "surrounding context"}
            ],
            "study_elements": [
                {"element": "study_phase", "value": "Phase II", "context": "context"}
            ],
            "regulatory_aspects": [
                {"requirement": "FDA approval", "status": "required", "context": "context"}
            ],
            "personnel": [
                {"role": "Principal Investigator", "responsibilities": "list of responsibilities"}
            ],
            "patient_criteria": {
                "inclusion": ["criteria1", "criteria2"],
                "exclusion": ["criteria1", "criteria2"]
            },
            "endpoints": {
                "primary": ["endpoint1"],
                "secondary": ["endpoint1", "endpoint2"]
            }
        }
        
        Text to analyze:
        {text}
        """
        
        self.triple_extraction_prompt = """
        Extract factual relationships from the following clinical text as subject-predicate-object triples.
        
        Focus on clinically relevant relationships such as:
        - Drug-condition relationships
        - Procedure-outcome relationships
        - Patient-eligibility relationships
        - Study-requirement relationships
        
        Return as JSON array:
        [
            {
                "subject": "subject entity",
                "predicate": "relationship type",
                "object": "object entity",
                "confidence": 0.95,
                "context": "original sentence or phrase"
            }
        ]
        
        Text: {text}
        """
    
    def generate_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate embeddings for document chunks."""
        try:
            texts = [chunk['content'] for chunk in chunks]
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i]
            
            return chunks
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def extract_clinical_entities(self, chunk_content: str) -> Dict[str, Any]:
        """Extract clinical entities from chunk content."""
        try:
            prompt = self.clinical_prompt.format(text=chunk_content)
            response = self.llm(prompt)
            
            # Parse JSON response
            try:
                entities = json.loads(response.strip())
                return entities
            except json.JSONDecodeError:
                # If JSON parsing fails, return empty structure
                self.logger.warning(f"Failed to parse LLM response as JSON: {response}")
                return {
                    "medical_entities": [],
                    "study_elements": [],
                    "regulatory_aspects": [],
                    "personnel": [],
                    "patient_criteria": {"inclusion": [], "exclusion": []},
                    "endpoints": {"primary": [], "secondary": []}
                }
        except Exception as e:
            self.logger.error(f"Error extracting clinical entities: {str(e)}")
            return {
                "medical_entities": [],
                "study_elements": [],
                "regulatory_aspects": [],
                "personnel": [],
                "patient_criteria": {"inclusion": [], "exclusion": []},
                "endpoints": {"primary": [], "secondary": []}
            }
    
    def extract_triples(self, chunk_content: str) -> List[Dict[str, Any]]:
        """Extract factual triples from chunk content."""
        try:
            prompt = self.triple_extraction_prompt.format(text=chunk_content)
            response = self.llm(prompt)
            
            # Parse JSON response
            try:
                triples = json.loads(response.strip())
                return triples if isinstance(triples, list) else []
            except json.JSONDecodeError:
                self.logger.warning(f"Failed to parse triple extraction response as JSON: {response}")
                return []
        except Exception as e:
            self.logger.error(f"Error extracting triples: {str(e)}")
            return []
    
    def process_chunk_with_llm(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single chunk with LLM for entity and triple extraction."""
        content = chunk['content']
        
        # Extract clinical entities
        entities = self.extract_clinical_entities(content)
        
        # Extract triples
        triples = self.extract_triples(content)
        
        # Calculate token usage for cost tracking
        token_count = len(content.split()) * 1.3  # Approximate token count
        
        return {
            **chunk,
            'clinical_entities': entities,
            'extracted_triples': triples,
            'token_usage': int(token_count),
            'processing_timestamp': self._get_current_timestamp()
        }
    
    def batch_process_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 5) -> List[Dict[str, Any]]:
        """Process chunks in batches to optimize LLM calls."""
        processed_chunks = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            for chunk in batch:
                processed_chunk = self.process_chunk_with_llm(chunk)
                processed_chunks.append(processed_chunk)
        
        return processed_chunks
    
    def semantic_search(self, query: str, chunk_embeddings: List[List[float]], chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on chunks using embeddings."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for i, chunk_embedding in enumerate(chunk_embeddings):
                similarity = self._cosine_similarity(query_embedding, chunk_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity and return top results
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, (chunk_idx, similarity) in enumerate(similarities[:top_k]):
                result = {
                    **chunks[chunk_idx],
                    'similarity_score': similarity,
                    'rank': i + 1
                }
                results.append(result)
            
            return results
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
    
    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def calculate_processing_cost(self, total_tokens: int, cost_per_token: float = 0.0001) -> float:
        """Calculate processing cost based on token usage."""
        return total_tokens * cost_per_token
