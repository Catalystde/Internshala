"""
UI Components Module for RAG Quote Retrieval System

This module contains all the UI components and rendering functions:
- Quote card rendering
- Search interface components
- Sidebar configuration
- Results display

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
from typing import Dict, Any, List

# Third-party imports
import streamlit as st

# Local imports
from .config import QUICK_SEARCHES, SEARCH_EXAMPLES, DEFAULT_SEARCH_PARAMS, MESSAGES

def render_quote_card(result: Dict[str, Any], show_scores: bool = False):
    """
    Render a quote card with modern, clean styling.
    
    This function creates beautiful HTML cards for displaying quotes
    with all relevant information and optional score details.
    
    Args:
        result (Dict[str, Any]): Quote result dictionary
        show_scores (bool): Whether to show detailed scoring information
    """
    # Extract data from result dictionary
    quote = result['quote']
    author = result['author']
    tags = result['tags']
    rank = result['rank']
    score = result['score']
    
    # Create HTML for the quote card
    card_html = f"""
    <div class="quote-card">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 0.5rem;">
            <div style="flex: 1; color: #b2bec3; font-size: 0.95rem;">Rank #{rank}</div>
            <div><span class="score-badge">Score: {score:.3f}</span></div>
        </div>
        <div class="quote-text">"{quote}"</div>
        <div class="quote-author">— {author}</div>
        {f'<div class="quote-tags">Tags: {", ".join(tags)}</div>' if tags else ''}
    </div>
    """
    
    # Render the HTML card
    st.markdown(card_html, unsafe_allow_html=True)
    
    # Show detailed scores if requested
    if show_scores:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Semantic Score", f"{result['semantic_score']:.3f}")
        with col2:
            st.metric("Additional Score", f"{result['additional_score']:.3f}")

def render_search_interface():
    """
    Render the main search interface components.
    
    This function creates the search input field, quick search buttons,
    and search examples for user guidance.
    """
    # Search examples and tips
    with st.expander("Search Examples"):
        st.markdown("**Try these example searches:**")
        for example in SEARCH_EXAMPLES:
            st.markdown(f"- `{example}`")
    
    # Search input field
    query = st.text_input(
        "Enter your search query:",
        placeholder="e.g., 'so many books so little time' or 'quotes about love'",
        help="Search for quotes by content, author, or theme"
    )
    
    # Quick search buttons for common queries
    st.markdown("**Quick Searches:**")
    cols = st.columns(len(QUICK_SEARCHES))
    
    for i, (label, search_query) in enumerate(QUICK_SEARCHES):
        with cols[i]:
            if st.button(label):
                st.session_state['query'] = search_query
    
    # Return the query (either from input or quick search)
    return query or st.session_state.get('query', '')

def render_sidebar_config():
    """
    Render the sidebar configuration components.
    
    This function creates the sidebar with file paths, search parameters,
    and system information.
    
    Returns:
        tuple: (artifacts_dir, model_path, top_k, show_scores)
    """
    st.sidebar.header("Configuration")
    
    # File paths configuration
    artifacts_dir = st.sidebar.text_input(
        "Artifacts Directory", 
        value="./pipeline_artifacts",
        help="Directory containing the saved pipeline artifacts"
    )
    
    model_path = st.sidebar.text_input(
        "Model Path", 
        value="./improved_quote_retriever",
        help="Path to the fine-tuned sentence transformer model"
    )
    
    # Search parameters configuration
    st.sidebar.header("Search Parameters")
    top_k = st.sidebar.slider("Number of Results", 1, 20, DEFAULT_SEARCH_PARAMS["top_k"])
    show_scores = st.sidebar.checkbox("Show Detailed Scores", value=DEFAULT_SEARCH_PARAMS["show_scores"])
    
    return artifacts_dir, model_path, top_k, show_scores

def render_system_info(metadata: Dict[str, Any]):
    """
    Render system information in the sidebar.
    
    Args:
        metadata (Dict[str, Any]): System metadata dictionary
    """
    if metadata:
        st.sidebar.header("System Info")
        st.sidebar.metric("Total Quotes", metadata.get('num_quotes', 'N/A'))
        st.sidebar.metric("Embedding Dimension", metadata.get('embedding_dimension', 'N/A'))
        if 'created_at' in metadata:
            st.sidebar.info(f"Created: {metadata['created_at'][:10]}")

def render_search_results(results: List[Dict[str, Any]], search_time: float, show_scores: bool):
    """
    Render search results with performance metrics.
    
    Args:
        results (List[Dict[str, Any]]): List of search results
        search_time (float): Time taken for search
        show_scores (bool): Whether to show detailed scores
    """
    if results:
        st.success(MESSAGES["search_success"].format(count=len(results), time=search_time))
        
        # Results header
        st.header("Search Results")
        
        # Display each result as a quote card
        for result in results:
            render_quote_card(result, show_scores=show_scores)
            
            # Add visual separator between results
            st.markdown("---")
    else:
        st.warning(MESSAGES["search_no_results"])

def render_footer():
    """
    Render the application footer with credits and information.
    """
    st.markdown("---")
    st.markdown(
        "Powered by Sentence Transformers and FAISS for intelligent quote retrieval"
    )

def render_rag_pipeline_integration(query: str, top_k: int):
    """
    Render the RAG pipeline integration section.
    
    Args:
        query (str): Current search query
        top_k (int): Number of results to retrieve
    """
    if st.button("Run RAG Pipeline"):
        with st.spinner("Running RAG pipeline..."):
            # Import here to avoid circular imports
            from rag_pipeline import build_faiss_index, retrieve_quotes, generate_answer
            
            # Build FAISS index and retrieve quotes
            index, df, model = build_faiss_index()
            results = retrieve_quotes(query, model, index, df, top_k=top_k)
            
            # Display retrieved quotes
            st.json(results)
            
            # Generate answer using Mistral LLM
            answer = generate_answer(query, results)
            st.markdown(f"**LLM Answer:**\n\n{answer}")
    
    st.markdown("---")
    st.markdown("For evaluation, see `rag_evaluation.py` and the generated CSV.") 