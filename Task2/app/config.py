"""
Configuration module for the RAG Quote Retrieval Streamlit App.

This module contains all the configuration settings, styling, and constants
used throughout the application.

Author: RAG System Developer
Date: 2024
"""

# Streamlit page configuration
PAGE_CONFIG = {
    "page_title": "RAG Quote Retrieval",
    "page_icon": "",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Custom CSS styling for the application
CUSTOM_CSS = """
<style>
    /* Main container styling */
    .main {
        padding-top: 1rem;
    }
    
    /* Alert styling */
    .stAlert {
        margin-top: 1rem;
    }
    
    /* Quote card styling for modern, clean appearance */
    .quote-card {
        background: #fff;
        padding: 2rem 1.5rem 1.5rem 1.5rem;
        border-radius: 14px;
        margin: 1.5rem 0;
        box-shadow: 0 4px 16px rgba(44,62,80,0.07), 0 1.5px 4px rgba(44,62,80,0.04);
        border: none;
        transition: box-shadow 0.2s;
    }
    
    /* Hover effect for quote cards */
    .quote-card:hover {
        box-shadow: 0 8px 32px rgba(44,62,80,0.12), 0 3px 8px rgba(44,62,80,0.08);
    }
    
    /* Quote text styling */
    .quote-text {
        font-size: 1.25rem;
        font-style: italic;
        color: #222f3e;
        margin-bottom: 1.2rem;
        font-family: 'Georgia', 'Times New Roman', Times, serif;
        line-height: 1.6;
    }
    
    /* Author styling */
    .quote-author {
        font-weight: 600;
        color: #3c4858;
        margin-bottom: 0.5rem;
        font-size: 1.05rem;
    }
    
    /* Tags styling */
    .quote-tags {
        color: #8395a7;
        font-size: 0.95rem;
        margin-bottom: 0.5rem;
    }
    
    /* Score badge styling */
    .score-badge {
        background: #f1f2f6;
        color: #576574;
        padding: 0.18rem 0.7rem;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 500;
        box-shadow: none;
        margin-left: 0.5rem;
    }
    
    /* Form control styling */
    .stSelectbox label {
        font-weight: bold;
        color: #2c3e50;
    }
    
    .stSlider label {
        font-weight: bold;
        color: #2c3e50;
    }
</style>
"""

# Default file paths
DEFAULT_PATHS = {
    "artifacts_dir": "./pipeline_artifacts",
    "model_path": "./improved_quote_retriever",
    "data_file": "quotes_data.csv",
    "index_file": "faiss_index.bin",
    "metadata_file": "metadata.json"
}

# Search examples for user guidance
SEARCH_EXAMPLES = [
    "so many books so little time",
    "quotes about love",
    "inspirational quotes",
    "Marcus Tullius Cicero quotes",
    "quotes by Frank Zappa",
    "wisdom quotes"
]

# Quick search options
QUICK_SEARCHES = [
    ("Book Quotes", "quotes about books"),
    ("Love Quotes", "quotes about love"),
    ("Inspirational", "inspirational quotes"),
    ("Wisdom", "wisdom quotes")
]

# Default search parameters
DEFAULT_SEARCH_PARAMS = {
    "top_k": 5,
    "show_scores": False
}

# Success and error messages
MESSAGES = {
    "model_loaded": "Loaded fine-tuned model from {path}",
    "model_fallback": "Using base model (fine-tuned model not found)",
    "model_error": "Error loading model: {error}",
    "data_loaded": "Loaded {count} quotes",
    "data_error": "Error loading data: {error}",
    "index_loaded": "Loaded FAISS index with {count} vectors",
    "index_error": "Error loading FAISS index: {error}",
    "metadata_error": "Error loading metadata: {error}",
    "search_success": "Found {count} results in {time:.2f} seconds",
    "search_no_results": "No results found. Try a different search query.",
    "search_error": "Search error: {error}",
    "artifacts_not_found": "Artifacts directory not found: {path}",
    "artifacts_help": "Please run the pipeline artifact creation script first!"
} 