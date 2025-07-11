
import numpy as np
from typing import List, Dict, Any
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import os

class EmbeddingManager:
    """Manages document embeddings using TF-IDF vectorization."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=2
        )
        self.embeddings_cache = {}
        self.fitted = False
        
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate TF-IDF embeddings for a list of texts."""
        try:
            if not self.fitted:
                # Fit the vectorizer on the texts
                embeddings = self.vectorizer.fit_transform(texts)
                self.fitted = True
            else:
                # Transform using the already fitted vectorizer
                embeddings = self.vectorizer.transform(texts)
            
            return embeddings.toarray()
            
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        if not self.fitted:
            # If not fitted, fit on this single text (not ideal but works)
            embedding = self.vectorizer.fit_transform([text])
            self.fitted = True
        else:
            embedding = self.vectorizer.transform([text])
        
        return embedding.toarray()[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings."""
        try:
            similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            return float(similarity)
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {str(e)}")
            return 0.0
    
    def find_similar_chunks(self, query_embedding: np.ndarray, 
                           chunk_embeddings: List[np.ndarray], 
                           top_k: int = 5) -> List[Dict[str, Any]]:
        """Find most similar chunks to a query embedding."""
        try:
            similarities = []
            
            for idx, chunk_embedding in enumerate(chunk_embeddings):
                similarity = self.calculate_similarity(query_embedding, chunk_embedding)
                similarities.append({
                    'chunk_index': idx,
                    'similarity_score': similarity
                })
            
            # Sort by similarity score in descending order
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error finding similar chunks: {str(e)}")
            return []
    
    def semantic_search(self, query: str, documents: List[Dict[str, Any]], 
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on documents."""
        try:
            # Extract text content from documents
            doc_texts = [doc.get('content', '') for doc in documents]
            
            # Generate embeddings for all documents
            doc_embeddings = self.generate_embeddings(doc_texts)
            
            # Generate embedding for query
            query_embedding = self.generate_single_embedding(query)
            
            # Calculate similarities
            similarities = []
            for idx, doc_embedding in enumerate(doc_embeddings):
                similarity = self.calculate_similarity(query_embedding, doc_embedding)
                result = documents[idx].copy()
                result['similarity_score'] = similarity
                similarities.append(result)
            
            # Sort by similarity
            similarities.sort(key=lambda x: x['similarity_score'], reverse=True)
            
            return similarities[:top_k]
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """Save embeddings to file."""
        try:
            with open(file_path, 'wb') as f:
                pickle.dump({
                    'embeddings': embeddings,
                    'vectorizer': self.vectorizer
                }, f)
            self.logger.info(f"Embeddings saved to {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
    
    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Load embeddings from file."""
        try:
            if os.path.exists(file_path):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    self.vectorizer = data['vectorizer']
                    self.fitted = True
                    self.logger.info(f"Embeddings loaded from {file_path}")
                    return data['embeddings']
            else:
                self.logger.warning(f"Embeddings file not found: {file_path}")
                return np.array([])
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            return np.array([])
    
    def batch_process_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process chunks in batches to generate embeddings."""
        try:
            texts = [chunk.get('content', '') for chunk in chunks]
            embeddings = self.generate_embeddings(texts)
            
            # Add embeddings to chunks
            for i, chunk in enumerate(chunks):
                chunk['embedding'] = embeddings[i].tolist()  # Convert to list for JSON serialization
            
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error in batch processing chunks: {str(e)}")
            return chunks
