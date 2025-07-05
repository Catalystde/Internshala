"""
Search Engine Module for RAG Quote Retrieval System

This module contains the core search functionality including:
- QuoteSearchEngine class for hybrid search
- Semantic similarity search using FAISS
- Keyword matching and scoring
- Result ranking and filtering

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
from typing import List, Dict, Any

# Third-party imports
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# Local imports
from .config import MESSAGES

class QuoteSearchEngine:
    """
    Main quote search engine class that combines semantic and keyword search.
    
    This class implements advanced search functionality that combines:
    - Semantic similarity using sentence transformers
    - Keyword matching for exact matches
    - Author and tag-based filtering
    - Hybrid scoring for optimal results
    
    Attributes:
        model (SentenceTransformer): The embedding model
        data (pd.DataFrame): The quotes dataset
        faiss_index (faiss.Index): The FAISS index for similarity search
    """
    
    def __init__(self, model, data, faiss_index):
        """
        Initialize the search engine with model, data, and index.
        
        Args:
            model (SentenceTransformer): The embedding model
            data (pd.DataFrame): The quotes dataset
            faiss_index (faiss.Index): The FAISS index
        """
        self.model = model
        self.data = data
        self.faiss_index = faiss_index
    
    def search_quotes(self, query: str, top_k: int = 5, boost_exact_match: bool = True) -> List[Dict[str, Any]]:
        """
        Search for quotes using semantic similarity and keyword matching.
        
        This function implements a hybrid search approach that combines:
        1. Semantic similarity using the FAISS index
        2. Keyword matching for exact matches
        3. Author and tag-based scoring
        4. Hybrid ranking for optimal results
        
        Args:
            query (str): The search query
            top_k (int): Number of results to return
            boost_exact_match (bool): Whether to boost exact keyword matches
            
        Returns:
            List[Dict[str, Any]]: List of search results with scores and metadata
        """
        # Handle empty queries
        if not query.strip():
            return []
        
        try:
            # Encode the query using the sentence transformer model
            # This converts the text query into a vector representation
            query_embedding = self.model.encode([query])
            
            # Normalize the query embedding for cosine similarity
            # This ensures consistent distance calculations
            faiss.normalize_L2(query_embedding)
            
            # Search with higher k for re-ranking
            # We retrieve more results than needed for better re-ranking
            search_k = min(top_k * 3, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_embedding.astype('float32'), search_k)
            
            # Prepare results with additional scoring
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.data):
                    quote_data = self.data.iloc[idx]
                    
                    # Calculate additional score based on keyword matching
                    additional_score = self._calculate_additional_score(query, quote_data)
                    
                    # Combine semantic score with additional scoring factors
                    final_score = float(score) + additional_score
                    
                    # Create result dictionary with all relevant information
                    results.append({
                        'rank': i + 1,                    # Result rank
                        'score': final_score,              # Combined score
                        'semantic_score': float(score),    # Original semantic score
                        'additional_score': additional_score, # Additional scoring factors
                        'quote': quote_data['quote'],      # Quote text
                        'author': quote_data['author'],    # Quote author
                        'tags': quote_data['tags'] if isinstance(quote_data['tags'], list) else [], # Tags
                        'searchable_text': quote_data.get('searchable_text', '') # Additional searchable text
                    })
            
            # Sort results by final score (highest first)
            results.sort(key=lambda x: x['score'], reverse=True)
            
            # Update ranks after sorting
            for i, result in enumerate(results[:top_k]):
                result['rank'] = i + 1
            
            # Return top_k results
            return results[:top_k]
            
        except Exception as e:
            st.error(MESSAGES["search_error"].format(error=str(e)))
            return []
    
    def _calculate_additional_score(self, query: str, quote_data) -> float:
        """
        Calculate additional scoring factors beyond semantic similarity.
        
        This function implements various scoring factors:
        - Exact keyword matches
        - Author name matches
        - Tag relevance
        - Quote length considerations
        
        Args:
            query (str): The search query
            quote_data: Row from the quotes dataset
            
        Returns:
            float: Additional score to add to semantic score
        """
        score = 0.0
        query_lower = query.lower()
        
        # Exact keyword matches in quote text
        quote_lower = quote_data['quote'].lower()
        query_words = set(query_lower.split())
        quote_words = set(quote_lower.split())
        
        # Keyword overlap bonus
        # Reward quotes that contain exact words from the query
        overlap = len(query_words.intersection(quote_words))
        if overlap > 0:
            score += 0.1 * overlap
        
        # Author match bonus
        # Reward quotes by authors mentioned in the query
        if any(word in quote_data['author'].lower() for word in query_words):
            score += 0.2
        
        # Tag match bonus
        # Reward quotes with relevant tags
        if isinstance(quote_data['tags'], list):
            for tag in quote_data['tags']:
                if isinstance(tag, str) and tag.lower() in query_lower:
                    score += 0.15
        
        # Length penalty for very short quotes
        # Prefer longer, more substantial quotes
        if len(quote_data['quote']) < 50:
            score -= 0.05
        
        return score 