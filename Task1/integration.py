# Integration Module for Support Ticket Classification
# This module provides a unified prediction function that combines all the components:
# - Feature engineering
# - Entity extraction
# - Model prediction for both issue type and urgency level

import pandas as pd  # For data manipulation
from entity_extraction import extract_entities  # For extracting structured information from text

def predict_ticket(ticket_text: str, best_issue_model, best_urgency_model, features, le_issue, le_issue_type, le_product, le_failure, le_delay, le_urgency, product_list, complaint_keywords) -> dict:
    """
    Make predictions for a new ticket text using the trained models.
    
    Args:
        ticket_text (str): The raw ticket text to analyze
        best_issue_model: The trained model for issue type prediction
        best_urgency_model: The trained model for urgency level prediction
        features (list): List of feature names used by the models
        le_issue, le_issue_type, le_product, le_failure, le_delay, le_urgency: LabelEncoder objects
        product_list (list): List of known product names
        complaint_keywords (list): List of complaint keywords
        
    Returns:
        dict: Dictionary containing predictions and extracted entities:
            - 'predicted_issue_type': The predicted issue type
            - 'predicted_urgency_level': The predicted urgency level
            - 'entities': Extracted entities (products, dates, complaint keywords)
    """
    # ===== FEATURE EXTRACTION =====
    # Extract text-based features from the ticket text (same as in training)
    from feature_engineering import extract_text_features
    text_features = extract_text_features(ticket_text)
    
    # Handle missing values in extracted features
    failure_time = text_features['failure_time']
    if pd.isna(failure_time) or failure_time is None:
        failure_time = 'None'  # Default value for missing failure time
    
    delay_duration = text_features['delay_duration']
    if pd.isna(delay_duration) or delay_duration is None:
        delay_duration = 'None'  # Default value for missing delay duration
    
    # ===== PRODUCT IDENTIFICATION =====
    # Find which product the customer is referring to in the ticket text
    # If no product is mentioned, use the first product as default
    product = next((prod for prod in product_list if prod.lower() in ticket_text.lower()), product_list[0])
    product_encoded = le_product.transform([product])[0]  # Convert product name to number
    
    # ===== PREPARE INPUT DATA FOR MODELS =====
    # Create a DataFrame with all the features that our models expect
    # Note: issue_type_encoded is set to 0 initially (dummy value)
    input_data = pd.DataFrame({
        'issue_type_encoded': [0],  # Will be updated after issue type prediction
        'product_encoded': [product_encoded],
        'high_urgency_keywords': [text_features['high_urgency_keywords']],
        'support_contact': [text_features['support_contact']],
        'failure_time_encoded': [le_failure.transform([failure_time])[0]],  # Convert failure time to number
        'delay_duration_encoded': [le_delay.transform([delay_duration])[0]]  # Convert delay duration to number
    })
    
    # ===== PREDICT ISSUE TYPE =====
    # Use the trained issue type model to predict the issue type
    issue_pred = best_issue_model.predict(input_data[features])
    issue_type_label = le_issue_type.inverse_transform(issue_pred)[0]  # Convert prediction back to text label
    
    # ===== UPDATE ISSUE TYPE FOR URGENCY PREDICTION =====
    # Now that we have the predicted issue type, encode it for the urgency model
    issue_encoded = le_issue.transform([issue_type_label])[0]
    input_data['issue_type_encoded'] = issue_encoded  # Update with the actual predicted issue type
    
    # ===== PREDICT URGENCY LEVEL =====
    # Use the trained urgency model to predict the urgency level
    urgency_pred = best_urgency_model.predict(input_data[features])
    urgency_label = le_urgency.inverse_transform(urgency_pred)[0]  # Convert prediction back to text label
    
    # ===== EXTRACT ENTITIES =====
    # Extract additional structured information from the ticket text
    entities = extract_entities(ticket_text, product_list, complaint_keywords)
    
    # ===== RETURN RESULTS =====
    # Return all predictions and extracted information
    return {
        'predicted_issue_type': issue_type_label,      # e.g., "Product Defect", "Billing Problem"
        'predicted_urgency_level': urgency_label,      # e.g., "Low", "Medium", "High"
        'entities': entities                           # Dictionary with products, dates, complaint keywords
    } 