"""
RAG Evaluation Module for Quote Retrieval System

This module implements evaluation of the RAG pipeline using RAGAS (Retrieval-Augmented Generation Assessment).
It provides automated evaluation of retrieval quality, answer generation, and overall system performance.

Key Features:
- RAGAS integration for comprehensive RAG evaluation
- Mistral LLM integration for answer generation and evaluation
- Custom evaluation metrics and scoring
- Automated evaluation pipeline with retry mechanisms

"""

# Standard library imports
import pandas as pd
import os
import time

# RAG pipeline components for retrieval and indexing
from rag_pipeline import retrieve_quotes, build_faiss_index

# RAGAS library for RAG evaluation
from ragas import evaluate

# Environment variable management
from dotenv import load_dotenv

# Mistral AI client for LLM integration
from mistralai import Mistral
from mistralai.models import SDKError

# LangChain components for RAG integration
from langchain_mistralai import ChatMistralAI
from langchain_community.embeddings import HuggingFaceEmbeddings

# HuggingFace datasets for RAGAS compatibility
from datasets import Dataset

# Load environment variables (API keys, etc.)
load_dotenv()

# Get Mistral API key from environment variables
# This should be set in a .env file for security
api_key = os.environ["MISTRAL_API_KEY"]

# Initialize Mistral client for direct API calls
mistral_client = Mistral(api_key=api_key)

# Initialize LangChain Mistral LLM for RAGAS evaluation
# This provides a standardized interface for the LLM that RAGAS can use
mistral_llm = ChatMistralAI(
    api_key=api_key,
    model="mistral-large-latest",
    temperature=1.0
)

# Initialize HuggingFace embeddings using the fine-tuned model
# This provides embeddings that are optimized for the quotes domain
embeddings = HuggingFaceEmbeddings(model_name="improved_quote_retriever")

def generate_mistral_answer(query, contexts, retries=3):
    """
    Generate an answer using Mistral LLM based on provided contexts.
    
    This function is used by RAGAS to generate answers for evaluation.
    It creates a prompt from the query and retrieved contexts, then uses
    Mistral to generate a coherent answer.
    
    Args:
        query (str): The user query to answer
        contexts (list): List of context strings (retrieved quotes)
        retries (int): Number of retry attempts for API calls
        
    Returns:
        str: Generated answer from the LLM
        
    Raises:
        SDKError: If Mistral API calls fail after all retries
    """
    # Create context text from the provided contexts
    # This combines all retrieved quotes into a single context
    context_text = "\n".join(contexts)
    
    # Construct the prompt for the LLM
    # This provides the context and asks the LLM to answer the query
    prompt = f"Given the following quotes:\n{context_text}\n\nAnswer the following question:\n{query}"
    
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

def evaluate_rag_system_with_mistral(df, model, index, queries, ground_truths, top_k=5):
    """
    Evaluate the RAG system using RAGAS with Mistral LLM integration.
    
    This function implements a complete evaluation pipeline:
    1. Retrieves quotes for each query using the FAISS index
    2. Generates answers using Mistral LLM
    3. Evaluates the system using RAGAS metrics
    4. Saves evaluation results to CSV
    
    Args:
        df (pd.DataFrame): The quotes dataset
        model (SentenceTransformer): The embedding model
        index (faiss.Index): The FAISS index for similarity search
        queries (list): List of test queries
        ground_truths (list): List of ground truth answers
        top_k (int): Number of top results to retrieve
        
    Returns:
        dict: RAGAS evaluation scores
        
    Raises:
        Exception: If evaluation fails
    """
    # Prepare inputs for RAGAS evaluation
    ragas_inputs = []
    
    # Process each query and its ground truth
    for query, gt in zip(queries, ground_truths):
        # Retrieve relevant quotes for the query
        retrieved = retrieve_quotes(query, model, index, df, top_k=top_k)
        
        # Extract quote texts as contexts for answer generation
        contexts = [r['quote'] for r in retrieved]
        
        # Generate answer using Mistral LLM
        answer = generate_mistral_answer(query, contexts)
        
        # Create RAGAS input format
        # RAGAS expects specific column names for evaluation
        ragas_inputs.append({
            "user_input": query,        # The user's query
            "contexts": contexts,        # Retrieved contexts (quotes)
            "ground_truth": gt,          # Expected answer
            "answer": answer             # Generated answer
        })
    
    # Convert list to HuggingFace Dataset format
    # RAGAS requires a HuggingFace Dataset object for evaluation
    ragas_dataset = Dataset.from_list(ragas_inputs)
    
    # Run RAGAS evaluation with Mistral LLM and embeddings
    # This evaluates various aspects of the RAG system:
    # - Answer relevance
    # - Context relevance
    # - Faithfulness
    # - Answer correctness
    scores = evaluate(ragas_dataset, llm=mistral_llm, embeddings=embeddings)
    
    # Print evaluation results
    print(scores)
    
    # Save evaluation results to CSV for later analysis
    pd.DataFrame(scores).to_csv("ragas_evaluation.csv", index=False)
    
    # Return the evaluation scores
    return scores

# Main execution block - runs the evaluation when script is executed directly
if __name__ == "__main__":
    # Build the FAISS index and load necessary components
    # This creates the search infrastructure needed for evaluation
    index, df, model = build_faiss_index()
    
    # Define test queries for evaluation
    # These queries test different aspects of the RAG system
    queries = [
        "quotes about hope by Oscar Wilde",      # Author-specific query
        "inspirational quotes about life",       # Theme-based query
        "quotes about courage by women authors"  # Complex multi-criteria query
    ]
    
    # Define ground truth answers for evaluation
    # These are expected answers that the system should generate
    ground_truths = [
        "We are all in the gutter, but some of us are looking at the stars.",  # Example Oscar Wilde quote
        "The purpose of our lives is to be happy.",                            # Example life quote
        "I am not afraid of storms, for I am learning how to sail my ship."   # Example courage quote
    ]
    
    # Run the evaluation pipeline
    # This will test the RAG system and generate evaluation metrics
    evaluate_rag_system_with_mistral(df, model, index, queries, ground_truths, top_k=5) 