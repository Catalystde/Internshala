# Entity Extraction Module for Support Ticket Classification
# This module extracts structured information (entities) from ticket text
# such as product names, dates, and complaint keywords

import re  # For regular expressions (pattern matching)
from typing import Dict, List  # For type hints to make code more readable

def extract_entities(text: str, product_list: List[str], complaint_keywords: List[str]) -> Dict:
    """
    Extract structured entities from ticket text to help with analysis and classification.
    
    Args:
        text (str): The ticket text to analyze
        product_list (List[str]): List of known product names to search for
        complaint_keywords (List[str]): List of keywords that indicate complaints
        
    Returns:
        Dict: Dictionary containing extracted entities:
            - 'products': List of products mentioned in the text
            - 'dates': List of dates found in the text
            - 'complaint_keywords': List of complaint keywords found
    """
    entities = {}  # Dictionary to store all extracted entities
    
    # Extract product names mentioned in the text
    # Check if any product from our known list appears in the ticket text
    # Convert both to lowercase for case-insensitive matching
    found_products = [prod for prod in product_list if prod.lower() in text.lower()]
    entities['products'] = found_products
    
    # Extract dates using regex pattern
    # This pattern matches various date formats:
    # - MM/DD/YYYY or MM-DD-YYYY (e.g., "12/25/2023" or "12-25-2023")
    # - YYYY/MM/DD or YYYY-MM-DD (e.g., "2023/12/25" or "2023-12-25")
    # - MM/DD/YY or MM-DD-YY (e.g., "12/25/23" or "12-25-23")
    date_pattern = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2})\b'
    found_dates = re.findall(date_pattern, text)
    entities['dates'] = found_dates
    
    # Extract complaint keywords mentioned in the text
    # Check if any complaint keyword from our list appears in the ticket text
    # Convert to lowercase for case-insensitive matching
    found_keywords = [kw for kw in complaint_keywords if kw in text.lower()]
    entities['complaint_keywords'] = found_keywords
    
    return entities 