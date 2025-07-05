"""
Data Preparation Module for RAG Quote Retrieval System

This module handles the downloading, cleaning, and preprocessing of the quotes dataset
from HuggingFace. It performs essential data cleaning operations to ensure high-quality
input for the downstream machine learning pipeline.

Key Operations:
- Downloads the Abirate/english_quotes dataset from HuggingFace
- Performs text cleaning and normalization
- Handles missing values and data validation
- Saves cleaned data for use in the pipeline

Author: RAG System Developer
Date: 2024
"""

# Standard library imports for data manipulation
import pandas as pd
# HuggingFace datasets library for easy dataset access
from datasets import load_dataset

def download_and_prepare_quotes():
    """
    Downloads and prepares the quotes dataset for the RAG pipeline.
    
    This function performs the following operations:
    1. Downloads the Abirate/english_quotes dataset from HuggingFace
    2. Converts it to a pandas DataFrame for easier manipulation
    3. Applies text cleaning and normalization
    4. Handles missing values and data validation
    5. Saves the cleaned dataset to a CSV file
    
    Returns:
        pd.DataFrame: The cleaned and prepared quotes dataset
        
    Raises:
        Exception: If dataset download fails or processing errors occur
    """
    # Download the quotes dataset from HuggingFace using the datasets library
    # This provides a clean, standardized way to access the dataset
    dataset = load_dataset("Abirate/english_quotes", split="train")
    
    # Convert the HuggingFace dataset to a pandas DataFrame for easier manipulation
    # This allows us to use pandas' powerful data manipulation functions
    df = pd.DataFrame(dataset)
    
    # Basic text cleaning operations to improve data quality
    
    # Strip whitespace and convert quotes to lowercase for consistency
    # This helps with text matching and reduces vocabulary size
    df['quote'] = df['quote'].str.strip().str.lower()
    
    # Handle missing author values by filling with 'unknown' and normalize
    # This ensures we don't lose quotes due to missing author information
    df['author'] = df['author'].fillna('unknown').str.strip().str.lower()
    
    # Process tags: convert to lowercase and handle non-list values
    # Tags are important for semantic search and categorization
    df['tags'] = df['tags'].apply(lambda x: [t.lower() for t in x] if isinstance(x, list) else [])
    
    # Remove rows with missing quotes (essential for the pipeline)
    # Quotes are the core content, so we can't process rows without them
    df = df.dropna(subset=['quote'])
    
    # Filter out empty quotes (quotes with no content after cleaning)
    # This ensures we only process meaningful content
    df = df[df['quote'].str.len() > 0]
    
    # Reset the index to ensure clean, sequential indexing
    # This is important for downstream processing and FAISS indexing
    df = df.reset_index(drop=True)
    
    # Save the cleaned dataset to a CSV file for use in the pipeline
    # This provides a persistent, human-readable format for the data
    df.to_csv("quotes_clean.csv", index=False)
    
    # Return the cleaned DataFrame for immediate use if needed
    return df

# Main execution block - runs the data preparation when script is executed directly
if __name__ == "__main__":
    # Execute the data preparation pipeline
    # This will download, clean, and save the quotes dataset
    download_and_prepare_quotes() 