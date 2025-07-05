"""
Main Application Module for RAG Quote Retrieval System

This module serves as the main entry point for the Streamlit application.
It orchestrates all the components and provides the main application logic.

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
import os
import time

# Third-party imports
import streamlit as st

# Local imports
from .config import PAGE_CONFIG, CUSTOM_CSS, MESSAGES
from .data_loader import load_all_components
from .search_engine import QuoteSearchEngine
from .ui_components import (
    render_search_interface,
    render_sidebar_config,
    render_system_info,
    render_search_results,
    render_footer,
    render_rag_pipeline_integration
)

def setup_page():
    """
    Set up the Streamlit page configuration and styling.
    """
    # Configure Streamlit page settings
    st.set_page_config(**PAGE_CONFIG)
    
    # Apply custom CSS styling
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def check_artifacts(artifacts_dir: str):
    """
    Check if the required artifacts exist.
    
    Args:
        artifacts_dir (str): Path to artifacts directory
        
    Returns:
        bool: True if artifacts exist, False otherwise
    """
    if not os.path.exists(artifacts_dir):
        st.error(MESSAGES["artifacts_not_found"].format(path=artifacts_dir))
        st.info(MESSAGES["artifacts_help"])
        return False
    return True

def load_components(artifacts_dir: str, model_path: str):
    """
    Load all application components with error handling.
    
    Args:
        artifacts_dir (str): Directory containing pipeline artifacts
        model_path (str): Path to the fine-tuned model
        
    Returns:
        tuple: (search_engine, metadata) or (None, {}) on error
    """
    with st.spinner("Loading components..."):
        model, data, faiss_index, metadata = load_all_components(artifacts_dir, model_path)
        
        if model is None or data is None or faiss_index is None:
            return None, {}
        
        # Initialize the search engine
        search_engine = QuoteSearchEngine(model, data, faiss_index)
        return search_engine, metadata

def perform_search(search_engine, query: str, top_k: int):
    """
    Perform search using the search engine.
    
    Args:
        search_engine: The QuoteSearchEngine instance
        query (str): Search query
        top_k (int): Number of results to return
        
    Returns:
        tuple: (results, search_time) or ([], 0) on error
    """
    with st.spinner("Searching..."):
        # Measure search performance
        start_time = time.time()
        results = search_engine.search_quotes(query, top_k=top_k)
        search_time = time.time() - start_time
        
        return results, search_time

def main():
    """
    Main application function that orchestrates the entire application.
    
    This function:
    1. Sets up the page configuration
    2. Renders the sidebar and configuration
    3. Loads all necessary components
    4. Handles user interactions and search
    5. Displays results and additional features
    """
    # Set up page configuration and styling
    setup_page()
    
    # Title and description
    st.title("RAG Quote Retrieval System")
    st.markdown("*Discover inspiring quotes using advanced AI-powered search*")
    
    # Render sidebar configuration
    artifacts_dir, model_path, top_k, show_scores = render_sidebar_config()
    
    # Check if artifacts exist
    if not check_artifacts(artifacts_dir):
        st.stop()
    
    # Load all components
    search_engine, metadata = load_components(artifacts_dir, model_path)
    if search_engine is None:
        st.stop()
    
    # Display system information in sidebar
    render_system_info(metadata)
    
    # Main search interface
    st.header("Search Quotes")
    
    # Render search interface and get query
    query = render_search_interface()
    
    # Perform search when query is provided
    if query:
        results, search_time = perform_search(search_engine, query, top_k)
        
        # Display search results
        render_search_results(results, search_time, show_scores)
    
    # Render footer
    render_footer()
    
    # Render RAG pipeline integration
    render_rag_pipeline_integration(query, top_k)

# Main execution block
if __name__ == "__main__":
    main() 