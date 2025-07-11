
import os
import logging
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pickle

class EmbeddingManager:
    """Manages document embeddings for semantic search."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.logger = logging.getLogger(__name__)
        self.model_name = model_name
        
        # Initialize sentence transformer model
        try:
            self.model = SentenceTransformer(model_name)
            self.logger.info(f"Loaded embedding model: {model_name}")
        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {str(e)}")
            raise
        
        # Embedding cache
        self.embedding_cache = {}
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            self.logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text."""
        return self.generate_embeddings([text])[0]
    
    def add_chunks_embeddings(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add embeddings to chunks."""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.generate_embeddings(texts)
        
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i]
            chunk['embedding_model'] = self.model_name
        
        return chunks
    
    def semantic_search(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform semantic search on chunks."""
        try:
            # Generate query embedding
            query_embedding = self.generate_single_embedding(query)
            
            # Calculate similarities
            similarities = []
            for i, chunk in enumerate(chunks):
                if 'embedding' in chunk:
                    similarity = self._cosine_similarity(query_embedding, chunk['embedding'])
                    similarities.append((i, similarity))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = []
            for i, (chunk_idx, similarity) in enumerate(similarities[:top_k]):
                result = {
                    **chunks[chunk_idx],
                    'similarity_score': similarity,
                    'search_rank': i + 1
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in semantic search: {str(e)}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)
        
        dot_product = np.dot(vec1_np, vec2_np)
        norm_a = np.linalg.norm(vec1_np)
        norm_b = np.linalg.norm(vec2_np)
        
        if norm_a == 0 or norm_b == 0:
            return 0
        
        return dot_product / (norm_a * norm_b)
    
    def save_embeddings(self, embeddings: Dict[str, Any], filepath: str):
        """Save embeddings to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
            self.logger.info(f"Embeddings saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {str(e)}")
            raise
    
    def load_embeddings(self, filepath: str) -> Dict[str, Any]:
        """Load embeddings from file."""
        try:
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
            self.logger.info(f"Embeddings loaded from {filepath}")
            return embeddings
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {str(e)}")
            raise
    
    def cluster_embeddings(self, embeddings: List[List[float]], n_clusters: int = 5) -> List[int]:
        """Cluster embeddings using K-means."""
        try:
            from sklearn.cluster import KMeans
            
            embeddings_np = np.array(embeddings)
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(embeddings_np)
            
            return clusters.tolist()
        except ImportError:
            self.logger.warning("scikit-learn not available for clustering")
            return [0] * len(embeddings)
        except Exception as e:
            self.logger.error(f"Error clustering embeddings: {str(e)}")
            return [0] * len(embeddings)
    
    def find_similar_chunks(self, target_chunk: Dict[str, Any], all_chunks: List[Dict[str, Any]], 
                           threshold: float = 0.8) -> List[Dict[str, Any]]:
        """Find chunks similar to a target chunk."""
        if 'embedding' not in target_chunk:
            return []
        
        target_embedding = target_chunk['embedding']
        similar_chunks = []
        
        for chunk in all_chunks:
            if 'embedding' in chunk and chunk['chunk_id'] != target_chunk['chunk_id']:
                similarity = self._cosine_similarity(target_embedding, chunk['embedding'])
                if similarity >= threshold:
                    similar_chunks.append({
                        **chunk,
                        'similarity_score': similarity
                    })
        
        # Sort by similarity
        similar_chunks.sort(key=lambda x: x['similarity_score'], reverse=True)
        
        return similar_chunks
