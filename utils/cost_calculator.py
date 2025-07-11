
import logging
from typing import Dict, Any, List
from dataclasses import dataclass

@dataclass
class CostBreakdown:
    """Cost breakdown for processing operations."""
    token_cost: float
    embedding_cost: float
    total_cost: float
    tokens_used: int
    embeddings_generated: int
    
class CostCalculator:
    """Utility for calculating processing costs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Default pricing (can be configured)
        self.gpt_3_5_turbo_cost = 0.0015  # per 1K tokens
        self.gpt_4_cost = 0.03  # per 1K tokens
        self.embedding_cost = 0.0001  # per 1K tokens
        self.local_embedding_cost = 0.0  # Free for local models
        
        # Token estimation multipliers
        self.token_multiplier = 1.3  # Words to tokens ratio
        
    def estimate_chunk_processing_cost(self, chunks: List[Dict[str, Any]], 
                                     use_gpt4: bool = False,
                                     use_local_embeddings: bool = True) -> CostBreakdown:
        """Estimate cost for processing chunks."""
        
        # Calculate total tokens
        total_tokens = 0
        for chunk in chunks:
            word_count = chunk.get('word_count', 0)
            if word_count == 0:
                word_count = len(chunk.get('content', '').split())
            total_tokens += int(word_count * self.token_multiplier)
        
        # Calculate LLM cost
        llm_cost_per_1k = self.gpt_4_cost if use_gpt4 else self.gpt_3_5_turbo_cost
        token_cost = (total_tokens / 1000) * llm_cost_per_1k
        
        # Calculate embedding cost
        embedding_cost_per_1k = 0.0 if use_local_embeddings else self.embedding_cost
        embedding_cost = (total_tokens / 1000) * embedding_cost_per_1k
        
        total_cost = token_cost + embedding_cost
        
        return CostBreakdown(
            token_cost=token_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            tokens_used=total_tokens,
            embeddings_generated=len(chunks)
        )
    
    def calculate_actual_cost(self, token_usage: Dict[str, int], 
                            embedding_usage: Dict[str, int]) -> CostBreakdown:
        """Calculate actual cost based on usage."""
        
        # Token costs
        gpt_35_tokens = token_usage.get('gpt_3_5_turbo', 0)
        gpt_4_tokens = token_usage.get('gpt_4', 0)
        
        token_cost = (gpt_35_tokens / 1000) * self.gpt_3_5_turbo_cost
        token_cost += (gpt_4_tokens / 1000) * self.gpt_4_cost
        
        # Embedding costs
        openai_embeddings = embedding_usage.get('openai', 0)
        local_embeddings = embedding_usage.get('local', 0)
        
        embedding_cost = (openai_embeddings / 1000) * self.embedding_cost
        # Local embeddings are free
        
        total_cost = token_cost + embedding_cost
        
        return CostBreakdown(
            token_cost=token_cost,
            embedding_cost=embedding_cost,
            total_cost=total_cost,
            tokens_used=gpt_35_tokens + gpt_4_tokens,
            embeddings_generated=openai_embeddings + local_embeddings
        )
    
    def get_cost_summary(self, cost_breakdown: CostBreakdown) -> Dict[str, Any]:
        """Get formatted cost summary."""
        return {
            'total_cost': f"${cost_breakdown.total_cost:.4f}",
            'token_cost': f"${cost_breakdown.token_cost:.4f}",
            'embedding_cost': f"${cost_breakdown.embedding_cost:.4f}",
            'tokens_used': f"{cost_breakdown.tokens_used:,}",
            'embeddings_generated': f"{cost_breakdown.embeddings_generated:,}",
            'cost_per_token': f"${cost_breakdown.total_cost / cost_breakdown.tokens_used:.6f}" if cost_breakdown.tokens_used > 0 else "$0.000000",
            'cost_per_embedding': f"${cost_breakdown.total_cost / cost_breakdown.embeddings_generated:.6f}" if cost_breakdown.embeddings_generated > 0 else "$0.000000"
        }
    
    def estimate_document_cost(self, file_size_mb: float, 
                             estimated_pages: int = None) -> Dict[str, Any]:
        """Estimate cost based on document size."""
        
        # Rough estimation based on file size
        # Assume ~250 words per page, ~1000 words per MB
        if estimated_pages:
            estimated_words = estimated_pages * 250
        else:
            estimated_words = int(file_size_mb * 1000)
        
        estimated_tokens = int(estimated_words * self.token_multiplier)
        
        # Estimate chunks (assuming 1000 words per chunk)
        estimated_chunks = max(1, estimated_words // 1000)
        
        # Calculate costs
        token_cost = (estimated_tokens / 1000) * self.gpt_3_5_turbo_cost
        embedding_cost = (estimated_tokens / 1000) * self.embedding_cost
        total_cost = token_cost + embedding_cost
        
        return {
            'estimated_words': estimated_words,
            'estimated_tokens': estimated_tokens,
            'estimated_chunks': estimated_chunks,
            'token_cost': token_cost,
            'embedding_cost': embedding_cost,
            'total_cost': total_cost,
            'formatted_cost': f"${total_cost:.4f}"
        }
    
    def get_pricing_info(self) -> Dict[str, Any]:
        """Get current pricing information."""
        return {
            'gpt_3_5_turbo_per_1k_tokens': f"${self.gpt_3_5_turbo_cost:.4f}",
            'gpt_4_per_1k_tokens': f"${self.gpt_4_cost:.4f}",
            'embedding_per_1k_tokens': f"${self.embedding_cost:.4f}",
            'local_embedding_cost': "Free",
            'currency': 'USD'
        }
    
    def optimize_cost_suggestions(self, cost_breakdown: CostBreakdown) -> List[str]:
        """Provide cost optimization suggestions."""
        suggestions = []
        
        if cost_breakdown.total_cost > 1.0:
            suggestions.append("Consider using local embeddings to reduce costs")
        
        if cost_breakdown.token_cost > cost_breakdown.embedding_cost * 2:
            suggestions.append("Token costs are high - consider optimizing chunk sizes")
        
        if cost_breakdown.embeddings_generated > 1000:
            suggestions.append("Large number of embeddings - consider batch processing")
        
        if cost_breakdown.total_cost > 5.0:
            suggestions.append("High processing cost - consider splitting into smaller batches")
        
        return suggestions
