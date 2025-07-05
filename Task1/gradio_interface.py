# Gradio Interface Module for Support Ticket Classification
# This module creates a web-based user interface for interactive ticket classification
# Users can input ticket text and get predictions for issue type, urgency level, and extracted entities

import gradio as gr  # For creating web interfaces
from integration import predict_ticket  # Our main prediction function

def launch_gradio_interface(best_issue_model, best_urgency_model, features, le_issue, le_issue_type, le_product, le_failure, le_delay, le_urgency, product_list, complaint_keywords):
    """
    Launch a Gradio web interface for interactive ticket classification.
    
    Args:
        best_issue_model: The trained model for issue type prediction
        best_urgency_model: The trained model for urgency level prediction
        features (list): List of feature names used by the models
        le_issue, le_issue_type, le_product, le_failure, le_delay, le_urgency: LabelEncoder objects
        product_list (list): List of known product names
        complaint_keywords (list): List of complaint keywords
    """
    
    def gradio_predict(text):
        """
        Wrapper function for Gradio that takes text input and returns predictions.
        
        Args:
            text (str): The ticket text entered by the user
            
        Returns:
            tuple: (issue_type, urgency_level, entities)
        """
        # Call our main prediction function with all the trained models and encoders
        result = predict_ticket(text, best_issue_model, best_urgency_model, features, 
                              le_issue, le_issue_type, le_product, le_failure, le_delay, 
                              le_urgency, product_list, complaint_keywords)
        
        # Return the predictions in the format expected by Gradio
        return (result['predicted_issue_type'], result['predicted_urgency_level'], result['entities'])
    
    # Create the Gradio interface
    iface = gr.Interface(
        fn=gradio_predict,  # The function that processes the input
        inputs=gr.Textbox(lines=4, label="Ticket Text"),  # Input: text area for ticket text
        outputs=[
            gr.Textbox(label="Predicted Issue Type"),      # Output 1: predicted issue type
            gr.Textbox(label="Predicted Urgency Level"),   # Output 2: predicted urgency level
            gr.JSON(label="Extracted Entities")            # Output 3: extracted entities (JSON format)
        ],
        title="Ticket Issue/Urgency Predictor & Entity Extractor"  # Interface title
    )
    
    # Launch the web interface
    # This will open a local web server where users can interact with the model
    iface.launch() 