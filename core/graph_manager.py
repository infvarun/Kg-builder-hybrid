import os
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase
from datetime import datetime
import json

class GraphManager:
    """Manages Neo4j database operations for the clinical document converter."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.driver = None
        self.connect()

    def connect(self):
        """Establish connection to Neo4j database."""
        try:
            # Use environment variables for Neo4j connection
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD', 'password')

            self.driver = GraphDatabase.driver(uri, auth=(username, password))
            self.driver.verify_connectivity()
            self.logger.info("Connected to Neo4j database")
        except Exception as e:
            self.logger.warning(f"Neo4j not available: {str(e)}")
            self.driver = None  # Set to None to enable mock mode

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def initialize_database(self):
        """Initialize database with constraints and indexes."""
        if not self.driver:
            self.logger.info("Using mock database mode - Neo4j not available")
            return

        with self.driver.session() as session:
            # Create constraints
            constraints = [
                "CREATE CONSTRAINT document_name IF NOT EXISTS FOR (d:Document) REQUIRE d.name IS UNIQUE",
                "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
                "CREATE CONSTRAINT upload_id IF NOT EXISTS FOR (u:Upload) REQUIRE u.upload_id IS UNIQUE"
            ]

            for constraint in constraints:
                try:
                    session.run(constraint)
                except Exception as e:
                    self.logger.warning(f"Constraint creation failed: {str(e)}")

            # Create indexes
            indexes = [
                "CREATE INDEX chunk_page_idx IF NOT EXISTS FOR (c:Chunk) ON (c.page_number)",
                "CREATE INDEX document_upload_date_idx IF NOT EXISTS FOR (d:Document) ON (d.upload_date)",
                "CREATE INDEX chunk_content_idx IF NOT EXISTS FOR (c:Chunk) ON (c.content)"
            ]

            for index in indexes:
                try:
                    session.run(index)
                except Exception as e:
                    self.logger.warning(f"Index creation failed: {str(e)}")

    def save_document(self, document_data: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
        """Save document and chunks to Neo4j."""
        if not self.driver:
            self.logger.info("Mock mode: Document would be saved to Neo4j")
            return document_data['name']

        with self.driver.session() as session:
            # Create document node
            document_query = """
            CREATE (d:Document {
                name: $name,
                upload_date: $upload_date,
                file_path: $file_path,
                total_chunks: $total_chunks,
                processing_status: $processing_status,
                file_hash: $file_hash,
                metadata: $metadata
            })
            RETURN d.name as name
            """

            doc_result = session.run(document_query, {
                'name': document_data['name'],
                'upload_date': datetime.now().isoformat(),
                'file_path': document_data.get('file_path', ''),
                'total_chunks': len(chunks),
                'processing_status': 'processing',
                'file_hash': document_data.get('file_hash', ''),
                'metadata': json.dumps(document_data.get('metadata', {}))
            })

            document_name = doc_result.single()['name']

            # Create chunks and relationships
            for chunk in chunks:
                chunk_query = """
                MATCH (d:Document {name: $document_name})
                CREATE (c:Chunk {
                    chunk_id: $chunk_id,
                    content: $content,
                    page_number: $page_number,
                    paragraph_numbers: $paragraph_numbers,
                    word_count: $word_count,
                    char_count: $char_count,
                    chunk_type: $chunk_type,
                    table_id: $table_id
                })
                CREATE (d)-[:CONTAINS]->(c)
                """

                session.run(chunk_query, {
                    'document_name': document_name,
                    'chunk_id': chunk['chunk_id'],
                    'content': chunk['content'],
                    'page_number': chunk['page_number'],
                    'paragraph_numbers': chunk['paragraph_numbers'],
                    'word_count': chunk['word_count'],
                    'char_count': chunk['char_count'],
                    'chunk_type': chunk['chunk_type'],
                    'table_id': chunk.get('table_id', '')
                })

            # Update document status
            session.run(
                "MATCH (d:Document {name: $name}) SET d.processing_status = 'completed'",
                {'name': document_name}
            )

            return document_name

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from database."""
        if not self.driver:
            return [
                {
                    'name': 'Sample Clinical Study.pdf',
                    'upload_date': '2024-01-15T10:30:00',
                    'status': 'completed',
                    'total_chunks': 25,
                    'actual_chunks': 25,
                    'metadata': {'num_pages': 12}
                }
            ]

        with self.driver.session() as session:
            query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            RETURN d.name as name, d.upload_date as upload_date, 
                   d.processing_status as status, d.total_chunks as total_chunks,
                   d.metadata as metadata, count(c) as actual_chunks
            ORDER BY d.upload_date DESC
            """

            result = session.run(query)
            documents = []

            for record in result:
                metadata = json.loads(record['metadata']) if record['metadata'] else {}
                documents.append({
                    'name': record['name'],
                    'upload_date': record['upload_date'],
                    'status': record['status'],
                    'total_chunks': record['total_chunks'],
                    'actual_chunks': record['actual_chunks'],
                    'metadata': metadata
                })

            return documents

    def delete_document(self, document_name: str) -> bool:
        """Delete document and all related nodes."""
        with self.driver.session() as session:
            query = """
            MATCH (d:Document {name: $name})
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            OPTIONAL MATCH (c)-[:EXTRACTED_FROM]->(t:Triple)
            OPTIONAL MATCH (c)-[:REFERENCES]->(cite:Citation)
            DETACH DELETE d, c, t, cite
            """

            result = session.run(query, {'name': document_name})
            return result.consume().counters.nodes_deleted > 0

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.driver:
            return {
                'total_documents': 1,
                'total_chunks': 25,
                'total_triples': 50,
                'total_words': 5000
            }

        with self.driver.session() as session:
            stats_query = """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:CONTAINS]->(c:Chunk)
            OPTIONAL MATCH (c)-[:EXTRACTED_FROM]->(t:Triple)
            RETURN count(DISTINCT d) as total_documents,
                   count(c) as total_chunks,
                   count(t) as total_triples,
                   sum(c.word_count) as total_words
            """

            result = session.run(stats_query)
            record = result.single()

            return {
                'total_documents': record['total_documents'] or 0,
                'total_chunks': record['total_chunks'] or 0,
                'total_triples': record['total_triples'] or 0,
                'total_words': record['total_words'] or 0
            }

    def search_chunks(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search chunks by content."""
        if not self.driver:
            return [
                {
                    'document_name': 'Sample Clinical Study.pdf',
                    'chunk_id': 'chunk_001',
                    'content': f'Mock search result for query: {query}. This would contain relevant clinical study information.',
                    'page_number': 1,
                    'chunk_type': 'paragraph'
                }
            ]

        with self.driver.session() as session:
            search_query = """
            MATCH (d:Document)-[:CONTAINS]->(c:Chunk)
            WHERE c.content CONTAINS $query
            RETURN d.name as document_name, c.chunk_id as chunk_id,
                   c.content as content, c.page_number as page_number,
                   c.chunk_type as chunk_type
            ORDER BY c.page_number
            LIMIT $limit
            """

            result = session.run(search_query, {'query': query, 'limit': limit})
            chunks = []

            for record in result:
                chunks.append({
                    'document_name': record['document_name'],
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'page_number': record['page_number'],
                    'chunk_type': record['chunk_type']
                })

            return chunks

    def save_upload_metadata(self, upload_data: Dict[str, Any]) -> str:
        """Save upload metadata."""
        if not self.driver:
            self.logger.info("Mock mode: Upload metadata would be saved")
            return upload_data['upload_id']

        with self.driver.session() as session:
            query = """
            CREATE (u:Upload {
                upload_id: $upload_id,
                document_name: $document_name,
                upload_timestamp: $upload_timestamp,
                total_chunks: $total_chunks,
                tokens_used: $tokens_used,
                status: $status,
                processing_cost: $processing_cost
            })
            RETURN u.upload_id as upload_id
            """

            result = session.run(query, {
                'upload_id': upload_data['upload_id'],
                'document_name': upload_data['document_name'],
                'upload_timestamp': datetime.now().isoformat(),
                'total_chunks': upload_data['total_chunks'],
                'tokens_used': upload_data.get('tokens_used', 0),
                'status': upload_data['status'],
                'processing_cost': upload_data.get('processing_cost', 0.0)
            })

            return result.single()['upload_id']

    def get_document_chunks(self, document_name: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Get chunks for a specific document."""
        if not self.driver:
            return [
                {
                    'chunk_id': 'chunk_001',
                    'content': f'Mock content for {document_name}',
                    'page_number': 1,
                    'chunk_type': 'paragraph',
                    'word_count': 50,
                    'char_count': 300
                }
            ]

        with self.driver.session() as session:
            result = session.run("""
                MATCH (d:Document {name: $document_name})-[:CONTAINS]->(c:Chunk)
                RETURN c.chunk_id as chunk_id,
                       c.content as content,
                       c.page_number as page_number,
                       c.chunk_type as chunk_type,
                       c.word_count as word_count,
                       c.char_count as char_count
                ORDER BY c.page_number
                LIMIT $limit
            """, document_name=document_name, limit=limit)

            chunks = []
            for record in result:
                chunks.append({
                    'chunk_id': record['chunk_id'],
                    'content': record['content'],
                    'page_number': record['page_number'],
                    'chunk_type': record['chunk_type'],
                    'word_count': record['word_count'],
                    'char_count': record['char_count']
                })

            return chunks