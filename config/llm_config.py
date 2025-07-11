
import os
from typing import Dict, Any, Optional

class LLMConfig:
    """Configuration for LLM services."""
    
    def __init__(self):
        # OpenAI Configuration
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_model = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
        self.openai_temperature = float(os.getenv('OPENAI_TEMPERATURE', '0.1'))
        self.openai_max_tokens = int(os.getenv('OPENAI_MAX_TOKENS', '2000'))
        
        # Embedding Configuration
        self.embedding_model = os.getenv('EMBEDDING_MODEL', 'text-embedding-ada-002')
        self.local_embedding_model = os.getenv('LOCAL_EMBEDDING_MODEL', 'all-MiniLM-L6-v2')
        self.use_local_embeddings = os.getenv('USE_LOCAL_EMBEDDINGS', 'true').lower() == 'true'
        
        # Processing Configuration
        self.batch_size = int(os.getenv('LLM_BATCH_SIZE', '5'))
        self.max_retries = int(os.getenv('LLM_MAX_RETRIES', '3'))
        self.retry_delay = float(os.getenv('LLM_RETRY_DELAY', '1.0'))
        
        # Cost Configuration
        self.cost_per_token = float(os.getenv('COST_PER_TOKEN', '0.0001'))
        self.cost_per_embedding = float(os.getenv('COST_PER_EMBEDDING', '0.0001'))
        
        # Clinical Processing Configuration
        self.enable_clinical_extraction = os.getenv('ENABLE_CLINICAL_EXTRACTION', 'true').lower() == 'true'
        self.enable_triple_extraction = os.getenv('ENABLE_TRIPLE_EXTRACTION', 'true').lower() == 'true'
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    
    def get_openai_config(self) -> Dict[str, Any]:
        """Get OpenAI configuration."""
        return {
            'api_key': self.openai_api_key,
            'model': self.openai_model,
            'temperature': self.openai_temperature,
            'max_tokens': self.openai_max_tokens
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration."""
        return {
            'model': self.embedding_model if not self.use_local_embeddings else self.local_embedding_model,
            'use_local': self.use_local_embeddings,
            'api_key': self.openai_api_key
        }
    
    def get_processing_config(self) -> Dict[str, Any]:
        """Get processing configuration."""
        return {
            'batch_size': self.batch_size,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_clinical_extraction': self.enable_clinical_extraction,
            'enable_triple_extraction': self.enable_triple_extraction,
            'confidence_threshold': self.confidence_threshold
        }
    
    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost configuration."""
        return {
            'cost_per_token': self.cost_per_token,
            'cost_per_embedding': self.cost_per_embedding
        }
    
    def validate_config(self) -> bool:
        """Validate configuration."""
        if not self.use_local_embeddings and not self.openai_api_key:
            print("Warning: OpenAI API key not set and local embeddings disabled")
            return False
        
        return True
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Create configuration from environment variables."""
        return cls()
    
    def get_clinical_prompts(self) -> Dict[str, str]:
        """Get clinical processing prompts."""
        return {
            'clinical_extraction': """
            You are a clinical research expert. Extract key medical and clinical information from the following text.
            
            Focus on:
            1. Medical entities (procedures, medications, conditions, medical devices)
            2. Study phases, timelines, and protocols
            3. Regulatory requirements and compliance points
            4. Investigator roles and responsibilities
            5. Patient populations and inclusion/exclusion criteria
            6. Primary and secondary endpoints
            7. Data collection methods and schedules
            
            Return structured JSON with extracted information.
            """,
            
            'triple_extraction': """
            Extract factual relationships from the following clinical text as subject-predicate-object triples.
            
            Focus on clinically relevant relationships such as:
            - Drug-condition relationships
            - Procedure-outcome relationships
            - Patient-eligibility relationships
            - Study-requirement relationships
            
            Return as JSON array with confidence scores.
            """
        }
