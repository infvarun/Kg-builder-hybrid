import openai
import logging
from typing import List, Dict, Any, Optional
import json
import re

class LLMProcessor:
    """Handles LLM interactions for triple extraction and entity recognition."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        self.logger = logging.getLogger(__name__)
        self.client = openai.OpenAI(api_key=api_key) if api_key else None
        self.model = model

    def extract_triples(self, text: str) -> List[Dict[str, Any]]:
        """Extract subject-predicate-object triples from text."""
        if not self.client:
            self.logger.warning("OpenAI client not initialized - using mock triples")
            return self._generate_mock_triples(text)

        try:
            prompt = self._create_triple_extraction_prompt(text)

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at extracting structured information from clinical documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            result = response.choices[0].message.content
            return self._parse_triple_response(result)

        except Exception as e:
            self.logger.error(f"Error extracting triples: {str(e)}")
            return self._generate_mock_triples(text)

    def _create_triple_extraction_prompt(self, text: str) -> str:
        """Create a prompt for triple extraction optimized for clinical documents."""
        return f"""
        Extract key relationships from the following clinical IRT study design text as subject-predicate-object triples.
        Focus on:
        - Medical procedures and their relationships
        - Study phases and timelines
        - Regulatory requirements
        - Investigator roles and responsibilities
        - Medications and conditions

        Text: {text}

        Return the triples in JSON format as a list of objects with 'subject', 'predicate', 'object', and 'confidence' fields.
        Confidence should be a float between 0 and 1.

        Example format:
        [
            {{"subject": "Phase I Study", "predicate": "includes", "object": "safety assessment", "confidence": 0.9}},
            {{"subject": "Investigator", "predicate": "responsible_for", "object": "patient enrollment", "confidence": 0.8}}
        ]
        """

    def _parse_triple_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse the LLM response to extract triples."""
        try:
            # Try to extract JSON from the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                triples = json.loads(json_str)

                # Validate and clean triples
                valid_triples = []
                for triple in triples:
                    if all(key in triple for key in ['subject', 'predicate', 'object']):
                        triple['confidence'] = float(triple.get('confidence', 0.5))
                        valid_triples.append(triple)

                return valid_triples

        except Exception as e:
            self.logger.error(f"Error parsing triple response: {str(e)}")

        return []

    def _generate_mock_triples(self, text: str) -> List[Dict[str, Any]]:
        """Generate mock triples for testing when OpenAI is not available."""
        words = text.split()[:20]  # Take first 20 words

        mock_triples = []
        if len(words) >= 3:
            mock_triples.append({
                'subject': words[0],
                'predicate': 'relates_to',
                'object': words[-1],
                'confidence': 0.7
            })

        if 'study' in text.lower():
            mock_triples.append({
                'subject': 'Clinical Study',
                'predicate': 'contains',
                'object': 'protocol information',
                'confidence': 0.8
            })

        return mock_triples

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text."""
        if not self.client:
            return self._extract_basic_entities(text)

        try:
            prompt = f"""
            Extract medical and clinical entities from the following text.
            Categorize them as: PROCEDURE, MEDICATION, CONDITION, PERSON, ORGANIZATION, DATE, LOCATION

            Text: {text}

            Return as JSON list with 'entity', 'category', and 'confidence' fields.
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at medical named entity recognition."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1
            )

            result = response.choices[0].message.content
            return self._parse_entity_response(result)

        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return self._extract_basic_entities(text)

    def _parse_entity_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse entity extraction response."""
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                entities = json.loads(json_match.group())
                return entities
        except Exception as e:
            self.logger.error(f"Error parsing entity response: {str(e)}")

        return []

    def _extract_basic_entities(self, text: str) -> List[Dict[str, Any]]:
        """Basic entity extraction using patterns."""
        entities = []

        # Simple pattern matching for common clinical terms
        clinical_terms = ['study', 'trial', 'patient', 'protocol', 'phase', 'treatment', 'medication']

        for term in clinical_terms:
            if term.lower() in text.lower():
                entities.append({
                    'entity': term,
                    'category': 'PROCEDURE' if term in ['study', 'trial', 'protocol'] else 'GENERAL',
                    'confidence': 0.6
                })

        return entities

    def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the text."""
        if not self.client:
            # Return first few sentences as summary
            sentences = text.split('.')[:3]
            return '. '.join(sentences) + '.'

        try:
            prompt = f"""
            Summarize the following clinical document text in {max_length} characters or less.
            Focus on key study objectives, procedures, and requirements.

            Text: {text}
            """

            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at summarizing clinical documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )

            return response.choices[0].message.content[:max_length]

        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return text[:max_length] + "..."