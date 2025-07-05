"""
RAG Pipeline Module for Quote Retrieval System

This module implements the core Retrieval-Augmented Generation (RAG) pipeline components:
1. FAISS index building for fast vector similarity search
2. Quote retrieval using semantic embeddings
3. Answer generation using Mistral LLM
4. Error handling and retry mechanisms for API calls

The pipeline combines semantic search with large language model generation
to provide intelligent quote retrieval and answer generation.

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
import pandas as pd
import os
import time

# Numerical computing libraries
import numpy as np

# FAISS library for efficient vector similarity search
import faiss

# Sentence transformers for generating embeddings
from sentence_transformers import SentenceTransformer

# Environment variable management
from dotenv import load_dotenv

# Mistral AI client for LLM integration
from mistralai import Mistral
from mistralai.models import SDKError

# LangChain components for RAG integration
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI

# Load environment variables (API keys, etc.)
load_dotenv()

# Get Mistral API key from environment variables
# This should be set in a .env file for security
api_key = os.environ["MISTRAL_API_KEY"]

# Initialize Mistral client for direct API calls
mistral_client = Mistral(api_key=api_key)

# Initialize LangChain Mistral LLM for RAGAS evaluation
# This provides a standardized interface for the LLM
mistral_llm = ChatMistralAI(
    api_key=api_key,
    model="mistral-large-latest",
    temperature=1.0
)

# Initialize HuggingFace embeddings using the fine-tuned model
# This provides embeddings that are optimized for the quotes domain
embeddings = HuggingFaceEmbeddings(model_name="improved_quote_retriever")

def build_faiss_index(model_dir="improved_quote_retriever", data_path="./pipeline_artifacts/quotes_data.csv", index_path="faiss.index"):
    """
    Build a FAISS index for fast vector similarity search.
    
    This function creates a vector index from the quotes dataset using the fine-tuned
    sentence transformer model. The index enables fast similarity search for retrieval.
    
    Args:
        model_dir (str): Path to the fine-tuned sentence transformer model
        data_path (str): Path to the processed quotes CSV file
        index_path (str): Path to save the FAISS index
        
    Returns:
        tuple: (FAISS index, DataFrame, SentenceTransformer model)
        
    Raises:
        FileNotFoundError: If model or data files don't exist
        Exception: If index building fails
    """
    # Load the processed quotes data
    # This contains the cleaned quotes from the data preparation step
    df = pd.read_csv(data_path)
    
    # Load the fine-tuned sentence transformer model
    # This model has been optimized for the quotes domain
    model = SentenceTransformer(model_dir)
    
    # Generate embeddings for all quotes in the dataset
    # show_progress_bar=True: Shows progress during encoding
    # convert_to_numpy=True: Returns numpy arrays for FAISS compatibility
    embeddings = model.encode(df['quote'].tolist(), show_progress_bar=True, convert_to_numpy=True)
    
    # Create a FAISS index for L2 (Euclidean) distance
    # IndexFlatL2 is a simple but effective index for similarity search
    # The dimension is determined by the embedding model output size
    index = faiss.IndexFlatL2(embeddings.shape[1])
    
    # Add the embeddings to the FAISS index
    # This makes them searchable for similarity queries
    index.add(embeddings)
    
    # Save the FAISS index to disk for later use
    # This allows us to load the index without rebuilding it
    faiss.write_index(index, index_path)
    
    # Save the embeddings as numpy array for potential reuse
    # This can be useful for analysis or debugging
    np.save("embeddings.npy", embeddings)
    
    # Return the index, data, and model for immediate use
    return index, df, model

def retrieve_quotes(query, model, index, df, top_k=5):
    """
    Retrieve the most similar quotes for a given query.
    
    This function performs semantic search using the FAISS index to find
    the most relevant quotes for the input query.
    
    Args:
        query (str): The search query
        model (SentenceTransformer): The embedding model
        index (faiss.Index): The FAISS index for similarity search
        df (pd.DataFrame): The quotes dataset
        top_k (int): Number of top results to return
        
    Returns:
        list: List of dictionaries containing quote information
        
    Raises:
        Exception: If retrieval fails
    """
    # Encode the query using the same model used for building the index
    # This ensures the query and quotes are in the same embedding space
    query_emb = model.encode([query])
    
    # Search the FAISS index for the most similar embeddings
    # D: distances (lower is more similar)
    # I: indices of the most similar quotes in the original dataset
    D, I = index.search(np.array(query_emb, dtype=np.float32), top_k)
    
    # Prepare the results in a structured format
    results = []
    for idx in I[0]:  # I[0] contains the indices for the first (and only) query
        # Get the corresponding row from the original dataset
        row = df.iloc[idx]
        
        # Create a result dictionary with quote information
        results.append({
            "quote": row['quote'],      # The quote text
            "author": row['author'],    # The quote author
            "tags": row['tags']         # Associated tags
        })
    
    return results

def generate_answer(query, retrieved_quotes, retries=3):
    """
    Generate an answer using Mistral LLM based on retrieved quotes.
    
    This function implements the generation part of the RAG pipeline.
    It creates a context from retrieved quotes and uses Mistral to generate
    a coherent answer to the user's query.
    
    Args:
        query (str): The original user query
        retrieved_quotes (list): List of retrieved quote dictionaries
        retries (int): Number of retry attempts for API calls
        
    Returns:
        str: Generated answer from the LLM
        
    Raises:
        SDKError: If Mistral API calls fail after all retries
    """
    # Create context from retrieved quotes
    # Format: "quote" - author for each quote
    context = "\n".join([f'"{q["quote"]}" - {q["author"]}' for q in retrieved_quotes])
    
    # Construct the prompt for the LLM
    # This provides the context and asks the LLM to answer based on it
    prompt = f"Context:\n{context}\n\nAnswer the following question based on the above quotes:\n{query}"
    
    # Implement retry logic for API calls
    # This handles temporary API failures and rate limits
    for attempt in range(retries):
        try:
            # Call Mistral API to generate the answer
            response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract and return the generated answer
            return response.choices[0].message.content
            
        except SDKError as e:
            # Handle API errors (rate limits, network issues, etc.)
            print(f"Mistral API error: {e}. Retrying in 5 seconds...")
            
            # Wait before retrying to avoid overwhelming the API
            time.sleep(5)
    
    # If all retries fail, return a fallback message
    return "[Mistral API unavailable]"

# Main execution block - runs the pipeline when script is executed directly
if __name__ == "__main__":
    # Build the FAISS index for quote retrieval
    # This creates the search infrastructure
    build_faiss_index()
    
    # Note: The evaluation line below is commented out as it requires
    # additional setup and is handled in rag_evaluation.py
    # scores = evaluate(ragas_inputs, llm=mistral_llm, embeddings=embeddings) 