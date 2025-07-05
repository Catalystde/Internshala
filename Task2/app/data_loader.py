"""
Data Loader Module for RAG Quote Retrieval System

This module handles loading and caching of all necessary components:
- Sentence transformer models
- Quotes data
- FAISS index
- Metadata

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
import os
import json

# Third-party imports
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

# Local imports
from .config import MESSAGES

@st.cache_resource
def load_model(model_path: str):
    """
    Load the trained sentence transformer model with caching.
    
    This function uses Streamlit's caching to avoid reloading the model
    on every interaction, significantly improving performance.
    
    Args:
        model_path (str): Path to the fine-tuned model directory
        
    Returns:
        SentenceTransformer: Loaded model or None if loading fails
    """
    try:
        # Check if the fine-tuned model exists
        if os.path.exists(model_path):
            model = SentenceTransformer(model_path)
            st.success(MESSAGES["model_loaded"].format(path=model_path))
        else:
            # Fallback to base model if fine-tuned model not found
            model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            st.warning(MESSAGES["model_fallback"])
        return model
    except Exception as e:
        st.error(MESSAGES["model_error"].format(error=str(e)))
        return None

@st.cache_data
def load_data(data_path: str):
    """
    Load the processed quotes data with caching.
    
    This function loads the CSV data and converts the tags column
    back to proper list format for processing.
    
    Args:
        data_path (str): Path to the quotes CSV file
        
    Returns:
        pd.DataFrame: Loaded data or None if loading fails
    """
    try:
        data = pd.read_csv(data_path)
        
        # Convert tags column back to lists
        # The tags are stored as string representations of lists
        data['tags'] = data['tags'].apply(lambda x: eval(x) if isinstance(x, str) and x.startswith('[') else [])
        
        st.success(MESSAGES["data_loaded"].format(count=len(data)))
        return data
    except Exception as e:
        st.error(MESSAGES["data_error"].format(error=str(e)))
        return None

@st.cache_resource
def load_faiss_index(index_path: str):
    """
    Load the FAISS index with caching.
    
    This function loads the pre-built FAISS index for fast similarity search.
    
    Args:
        index_path (str): Path to the FAISS index file
        
    Returns:
        faiss.Index: Loaded index or None if loading fails
    """
    try:
        index = faiss.read_index(index_path)
        st.success(MESSAGES["index_loaded"].format(count=index.ntotal))
        return index
    except Exception as e:
        st.error(MESSAGES["index_error"].format(error=str(e)))
        return None

@st.cache_data
def load_metadata(metadata_path: str):
    """
    Load metadata information with caching.
    
    This function loads metadata about the system configuration
    and performance metrics.
    
    Args:
        metadata_path (str): Path to the metadata JSON file
        
    Returns:
        dict: Metadata dictionary or empty dict if loading fails
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        return metadata
    except Exception as e:
        st.error(MESSAGES["metadata_error"].format(error=str(e)))
        return {}

def load_all_components(artifacts_dir: str, model_path: str):
    """
    Load all necessary components for the application.
    
    This function loads the model, data, FAISS index, and metadata
    in a single call with proper error handling.
    
    Args:
        artifacts_dir (str): Directory containing pipeline artifacts
        model_path (str): Path to the fine-tuned model
        
    Returns:
        tuple: (model, data, faiss_index, metadata) or (None, None, None, {}) on error
    """
    # Load metadata for system information
    metadata_path = os.path.join(artifacts_dir, "metadata.json")
    metadata = load_metadata(metadata_path)
    
    # Load the fine-tuned model
    model = load_model(model_path)
    if model is None:
        return None, None, None, {}
    
    # Load the quotes data
    data_path = os.path.join(artifacts_dir, "quotes_data.csv")
    data = load_data(data_path)
    if data is None:
        return None, None, None, {}
    
    # Load the FAISS index
    index_path = os.path.join(artifacts_dir, "faiss_index.bin")
    faiss_index = load_faiss_index(index_path)
    if faiss_index is None:
        return None, None, None, {}
    
    return model, data, faiss_index, metadata 