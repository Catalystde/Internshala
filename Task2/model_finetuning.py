"""
Model Fine-tuning Module for RAG Quote Retrieval System

This module handles the fine-tuning of a SentenceTransformer model on the quotes dataset
to improve semantic search performance. It creates a custom dataset class and implements
the fine-tuning pipeline using the sentence-transformers library.

Key Features:
- Custom QuotesDataset class for handling quote data
- Fine-tuning with MultipleNegativesRankingLoss for better embeddings
- Model architecture with Transformer + Pooling layers
- Automatic model saving and loading

Author: RAG System Developer
Date: 2024
"""

# Standard library imports
import pandas as pd
import os

# Sentence transformers library for fine-tuning embedding models
from sentence_transformers import SentenceTransformer, InputExample, losses, models

# PyTorch utilities for data loading and training
from torch.utils.data import DataLoader, Dataset

class QuotesDataset(Dataset):
    """
    Custom PyTorch Dataset class for handling quotes data during fine-tuning.
    
    This class wraps the quotes data in a format suitable for the sentence-transformers
    training pipeline. It creates InputExample objects that the model can process.
    
    Attributes:
        examples (list): List of InputExample objects containing the quotes
    """
    
    def __init__(self, quotes):
        """
        Initialize the dataset with quotes data.
        
        Args:
            quotes (list): List of quote strings to be used for fine-tuning
        """
        # Convert each quote into an InputExample object
        # InputExample is the standard format expected by sentence-transformers
        # For unsupervised training, we use the same text as both positive examples
        self.examples = [InputExample(texts=[str(q)]) for q in quotes]
    
    def __len__(self):
        """
        Return the total number of examples in the dataset.
        
        Returns:
            int: Number of examples in the dataset
        """
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a specific example from the dataset by index.
        
        Args:
            idx (int): Index of the example to retrieve
            
        Returns:
            InputExample: The example at the specified index
        """
        return self.examples[idx]

def finetune_sentence_model(data_path="quotes_clean.csv", model_name="all-MiniLM-L6-v2", output_dir="finetuned_model"):
    """
    Fine-tune a SentenceTransformer model on the quotes dataset.
    
    This function implements the complete fine-tuning pipeline:
    1. Loads the cleaned quotes data
    2. Creates a custom dataset for training
    3. Sets up the model architecture (Transformer + Pooling)
    4. Configures the training loss function
    5. Trains the model and saves it
    
    Args:
        data_path (str): Path to the cleaned quotes CSV file
        model_name (str): Name of the base model to fine-tune
        output_dir (str): Directory to save the fine-tuned model
        
    Returns:
        str: Path to the saved fine-tuned model
        
    Raises:
        FileNotFoundError: If the data file doesn't exist
        Exception: If fine-tuning fails
    """
    # Check if model directory already exists to avoid re-training
    # This saves time and computational resources
    if os.path.exists(output_dir):
        print(f"Model directory '{output_dir}' already exists. Skipping fine-tuning.")
        return output_dir
    
    # Load the cleaned quotes data from CSV
    # This contains the preprocessed quotes from the data preparation step
    df = pd.read_csv(data_path)
    
    # Extract quotes as strings for training
    # Convert to string to handle any non-string data types
    quotes = df['quote'].astype(str).tolist()
    
    # Create the custom dataset for training
    # This wraps the quotes in the format expected by sentence-transformers
    train_dataset = QuotesDataset(quotes)
    
    # Create a DataLoader for batch processing during training
    # Batch size of 32 is a good balance between memory usage and training speed
    # Shuffle=True ensures random sampling for better training
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Set up the model architecture
    # Transformer layer: handles the text encoding (e.g., BERT, RoBERTa)
    word_embedding_model = models.Transformer(model_name)
    
    # Pooling layer: aggregates word embeddings into sentence embeddings
    # Uses the output dimension from the transformer model
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    
    # Combine transformer and pooling layers into a complete model
    # This creates a model that can convert sentences to embeddings
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    
    # Set up the training loss function
    # MultipleNegativesRankingLoss is good for unsupervised training
    # It treats each sentence as a positive example for itself
    train_loss = losses.MultipleNegativesRankingLoss(model)
    
    # Train the model using the sentence-transformers fit method
    # epochs=1: Single pass through the data (can be increased for better results)
    # warmup_steps=100: Gradually increase learning rate for first 100 steps
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    
    # Save the fine-tuned model to the specified directory
    # This creates a complete model that can be loaded later
    model.save(output_dir)
    
    # Return the path to the saved model
    return output_dir

# Main execution block - runs the fine-tuning when script is executed directly
if __name__ == "__main__":
    # Execute the fine-tuning pipeline
    # This will load the data, train the model, and save it
    finetune_sentence_model() 