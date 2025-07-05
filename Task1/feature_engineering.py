# Feature Engineering Module for Support Ticket Classification
# This module creates new features from the raw ticket data to improve model performance
# It extracts text-based features and encodes categorical variables

import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
import re            # For regular expressions (pattern matching in text)
from sklearn.preprocessing import LabelEncoder  # For converting categorical variables to numbers
from nltk.tokenize import word_tokenize        # For splitting text into words
from nltk.corpus import stopwords              # For removing common words like 'the', 'and', etc.

def extract_text_features(text, stop_words=None):
    """
    Extract structured features from ticket text to help predict urgency and issue type.
    
    Args:
        text (str): The ticket text to analyze
        stop_words (set): Set of words to ignore (common words like 'the', 'and')
        
    Returns:
        pd.Series: A series containing extracted features
    """
    # Handle missing text by converting to empty string
    if pd.isna(text):
        text = ''
    
    # Convert to lowercase for consistent matching
    text = text.lower()
    
    # Set default stop words if none provided
    if stop_words is None:
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    # Split text into individual words (tokens)
    try:
        tokens = word_tokenize(text)  # Use NLTK tokenizer if available
    except:
        # Fallback to regex if NLTK fails
        tokens = re.findall(r'\b\w+\b', text)  # Find all word boundaries
    
    # Remove stop words (common words that don't add meaning)
    tokens = [t for t in tokens if t not in stop_words]
    
    # Define keywords that indicate high urgency issues
    high_urgency_keywords = ['stopped working', 'defective', 'broken', 'malfunction', 'no response', 
                             'charged twice', 'overbilled', 'urgent help', 'blocked', 'reset required']
    
    # Check if customer mentioned contacting support without response
    support_contact = 1 if 'contacted support but got no response' in text else 0
    
    # Check if any high urgency keywords are present
    high_urgency = 1 if any(kw in text for kw in high_urgency_keywords) else 0
    
    # Extract failure time information using regex
    failure_time = None
    # Look for pattern like "stopped working after just 5 days"
    match = re.search(r'stopped working after just (\d+) days', text)
    if match:
        days = int(match.group(1))  # Extract the number of days
        # Categorize failure time based on how quickly the product failed
        if days <= 3:
            failure_time = 'Short'      # Failed very quickly (≤3 days)
        elif days <= 7:
            failure_time = 'Medium'     # Failed moderately quickly (4-7 days)
        else:
            failure_time = 'Long'       # Failed after longer use (>7 days)
    
    # Extract delay duration information using regex
    delay_duration = None
    # Look for pattern like "5 days late"
    match = re.search(r'(\d+) days late', text)
    if match:
        days = int(match.group(1))  # Extract the number of days
        # Categorize delay duration based on how late the delivery was
        if days <= 5:
            delay_duration = 'Short'    # Short delay (≤5 days)
        elif days <= 10:
            delay_duration = 'Medium'   # Medium delay (6-10 days)
        else:
            delay_duration = 'Long'     # Long delay (>10 days)
    
    # Return all extracted features as a pandas Series
    return pd.Series({
        'high_urgency_keywords': high_urgency,  # Binary: 1 if urgent keywords found
        'support_contact': support_contact,      # Binary: 1 if support contacted without response
        'failure_time': failure_time,            # Categorical: Short/Medium/Long/None
        'delay_duration': delay_duration         # Categorical: Short/Medium/Long/None
    })

def add_features_and_encoders(df):
    """
    Add engineered features to the DataFrame and encode categorical variables.
    
    Args:
        df (pd.DataFrame): DataFrame containing ticket data
        
    Returns:
        tuple: (processed_df, le_issue, le_product, le_failure, le_delay)
               - processed_df: DataFrame with new features
               - le_*: LabelEncoder objects for converting categories to numbers
    """
    # Create LabelEncoder objects for categorical variables
    le_issue = LabelEncoder()   # For issue_type column
    le_product = LabelEncoder()  # For product column
    
    # Convert categorical variables to numerical values
    # fit_transform() learns the categories and converts them to numbers (0, 1, 2, etc.)
    df['issue_type_encoded'] = le_issue.fit_transform(df['issue_type'])
    df['product_encoded'] = le_product.fit_transform(df['product'])
    
    # Extract text-based features from ticket text
    try:
        # Try to get English stop words from NLTK
        stop_words = set(stopwords.words('english'))
    except:
        # Fallback to manual stop words if NLTK fails
        stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    
    # Apply text feature extraction to each ticket
    # This creates new columns: high_urgency_keywords, support_contact, failure_time, delay_duration
    text_features = df['ticket_text'].apply(lambda x: extract_text_features(x, stop_words))
    
    # Combine original DataFrame with new text features
    df = pd.concat([df, text_features], axis=1)
    
    # Handle missing values in the new categorical features
    df['failure_time'] = df['failure_time'].fillna('None')      # Replace NaN with 'None'
    df['delay_duration'] = df['delay_duration'].fillna('None')  # Replace NaN with 'None'
    
    # Create encoders for the new categorical features
    le_failure = LabelEncoder()  # For failure_time column
    le_delay = LabelEncoder()    # For delay_duration column
    
    # Convert the new categorical features to numerical values
    df['failure_time_encoded'] = le_failure.fit_transform(df['failure_time'])
    df['delay_duration_encoded'] = le_delay.fit_transform(df['delay_duration'])
    
    # Return the processed DataFrame and all encoders for later use
    return df, le_issue, le_product, le_failure, le_delay 