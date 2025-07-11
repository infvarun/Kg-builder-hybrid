
import os
from typing import Dict, Any

class Neo4jConfig:
    """Configuration for Neo4j database connection."""
    
    def __init__(self):
        self.uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
        self.username = os.getenv('NEO4J_USERNAME', 'neo4j')
        self.password = os.getenv('NEO4J_PASSWORD', 'password')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')
        
        # Connection pool settings
        self.max_connection_lifetime = 3600  # 1 hour
        self.max_connection_pool_size = 50
        self.connection_acquisition_timeout = 60  # seconds
        
        # Retry settings
        self.max_retry_time = 30  # seconds
        self.initial_retry_delay = 1  # second
        self.multiplier = 2.0
        self.jitter_factor = 0.2
    
    def get_connection_config(self) -> Dict[str, Any]:
        """Get connection configuration dictionary."""
        return {
            'uri': self.uri,
            'auth': (self.username, self.password),
            'max_connection_lifetime': self.max_connection_lifetime,
            'max_connection_pool_size': self.max_connection_pool_size,
            'connection_acquisition_timeout': self.connection_acquisition_timeout,
            'max_retry_time': self.max_retry_time,
            'initial_retry_delay': self.initial_retry_delay,
            'multiplier': self.multiplier,
            'jitter_factor': self.jitter_factor
        }
    
    def validate_config(self) -> bool:
        """Validate configuration settings."""
        required_vars = ['NEO4J_URI', 'NEO4J_USERNAME', 'NEO4J_PASSWORD']
        
        for var in required_vars:
            if not os.getenv(var):
                print(f"Warning: {var} environment variable not set")
                return False
        
        return True
    
    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create configuration from environment variables."""
        return cls()
    
    def get_bolt_url(self) -> str:
        """Get formatted Bolt URL."""
        return self.uri
    
    def get_http_url(self) -> str:
        """Get HTTP URL for Neo4j browser."""
        return self.uri.replace('bolt://', 'http://').replace(':7687', ':7474')
