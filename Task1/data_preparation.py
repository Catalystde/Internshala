# Data Preparation Module for Support Ticket Classification
# This module handles loading, cleaning, and preprocessing of support ticket data
# before it's used for feature engineering and model training

import pandas as pd  # For data manipulation and analysis
import numpy as np   # For numerical operations

# List of products available in the system - used for entity extraction
# These help identify which product a customer is referring to in their ticket
PRODUCT_LIST = [
    'SmartWatch V2', 'UltraClean Vacuum', 'SoundWave 300', 'EcoBreeze AC', 'PowerMax Battery', 'PhotoSnap Cam'
]

# Keywords that indicate a complaint or urgent issue
# These help classify urgency levels and identify problematic tickets
COMPLAINT_KEYWORDS = ['broken', 'defective', 'late', 'error', 'missing', 'stopped working', 
                     'malfunction', 'no response', 'overbilled', 'charged twice']

def impute_urgency_by_issue(df):
    """
    Fill missing urgency levels based on the most common urgency for each issue type.
    
    Args:
        df (pd.DataFrame): DataFrame containing issue_type and urgency_level columns
        
    Returns:
        pd.DataFrame: DataFrame with missing urgency levels filled
    """
    # Group by issue_type and find the most common urgency level for each issue type
    # mode() returns the most frequent value, [0] gets the first value if there are ties
    urgency_by_issue = df.groupby('issue_type')['urgency_level'].agg(
        lambda x: x.mode()[0] if not x.mode().empty else 'Medium'
    )
    
    # Apply the mapping: if urgency_level is missing, use the most common urgency for that issue type
    # Otherwise, keep the existing urgency level
    df['urgency_level'] = df.apply(
        lambda row: urgency_by_issue[row['issue_type']] if pd.isna(row['urgency_level']) else row['urgency_level'],
        axis=1  # Apply function to each row
    )
    return df

def impute_issue_type(row):
    """
    Fill missing issue types by analyzing the ticket text content.
    
    Args:
        row (pd.Series): A single row from the DataFrame containing ticket_text and issue_type
        
    Returns:
        str: Predicted issue type based on text content
    """
    # If issue_type is missing, analyze the ticket text to predict the issue type
    if pd.isna(row['issue_type']):
        text = str(row['ticket_text']).lower()  # Convert to lowercase for case-insensitive matching
        
        # Check for specific keywords to categorize the issue type
        if 'stopped working' in text or 'defective' in text or 'malfunction' in text:
            return 'Product Defect'
        elif 'wrong product' in text or 'order mixed up' in text:
            return 'Wrong Item'
        elif 'log in' in text or 'account' in text:
            return 'Account Access'
        elif 'payment' in text or 'billed' in text:
            return 'Billing Problem'
        elif 'late' in text or 'not here' in text:
            return 'Late Delivery'
        elif 'warranty' in text or 'available' in text:
            return 'General Inquiry'
        elif 'setup fails' in text or 'installation' in text:
            return 'Installation Issue'
        else:
            return 'Unknown'  # Default category if no keywords match
    return row['issue_type']  # Return existing issue_type if not missing

def load_and_prepare_data(file_path):
    """
    Main function to load and prepare the support ticket data.
    
    Args:
        file_path (str): Path to the Excel file containing the ticket data
        
    Returns:
        pd.DataFrame: Cleaned and preprocessed DataFrame ready for feature engineering
    """
    # Load the data from Excel file
    df = pd.read_excel(file_path)
    
    # Fill missing issue types by analyzing ticket text
    df['issue_type'] = df.apply(impute_issue_type, axis=1)
    
    # Fill missing urgency levels based on issue type patterns
    df = impute_urgency_by_issue(df)
    
    return df 