# Main Application Module for Support Ticket Classification
# This is the orchestrator that runs the entire pipeline:
# 1. Data preparation and cleaning
# 2. Feature engineering
# 3. Model training and selection
# 4. Launching the web interface

# Import all the necessary functions from our modular components
from data_preparation import load_and_prepare_data, PRODUCT_LIST, COMPLAINT_KEYWORDS
from feature_engineering import add_features_and_encoders
from multi_task_learning import train_and_select_models
from gradio_interface import launch_gradio_interface

# ===== STEP 1: DATA PREPARATION =====
# Load and clean the raw ticket data from the Excel file
file_path = 'C:/Users/Ankit/Desktop/Internshala/ai_dev_assignment_tickets_complex_1000.xls'
df = load_and_prepare_data(file_path)
# This step:
# - Loads the Excel file
# - Fills missing issue types by analyzing ticket text
# - Fills missing urgency levels based on issue type patterns

# ===== STEP 2: FEATURE ENGINEERING =====
# Add new features and encode categorical variables for model training
processed_df, le_issue, le_product, le_failure, le_delay = add_features_and_encoders(df)
# This step:
# - Extracts text-based features (urgency keywords, support contact, failure time, delay duration)
# - Encodes categorical variables (issue_type, product, failure_time, delay_duration) to numbers
# - Returns the processed DataFrame and all the fitted encoders

# ===== STEP 3: MULTI-TASK LEARNING (MODEL TRAINING/SELECTION) =====
# Import LabelEncoder for encoding target variables
from sklearn.preprocessing import LabelEncoder

# Create encoders for the target variables (what we want to predict)
le_urgency = LabelEncoder()      # For encoding urgency levels (Low, Medium, High)
le_issue_type = LabelEncoder()   # For encoding issue types (Product Defect, Billing Problem, etc.)

# Fit the issue_type encoder to ensure it knows all possible issue types
le_issue_type.fit(processed_df['issue_type'])

# Train and select the best models for both prediction tasks
best_issue_model, best_urgency_model, features, X, le_urgency, le_issue, le_product, le_failure, le_delay = train_and_select_models(
    processed_df, le_issue, le_product, le_failure, le_delay, le_urgency
)
# This step:
# - Splits data into training and testing sets
# - Trains multiple models (Logistic Regression, Random Forest, SVM, KNN, Gradient Boosting, Naive Bayes)
# - Compares their performance using accuracy and F1 score
# - Selects the best model for each task (issue type prediction and urgency level prediction)
# - Saves detailed model comparison report to 'model_report.txt'

# ===== STEP 4: LAUNCH GRADIO INTERFACE =====
# Launch the web interface for interactive predictions
launch_gradio_interface(
    best_issue_model, best_urgency_model, features, le_issue, le_issue_type, 
    le_product, le_failure, le_delay, le_urgency, PRODUCT_LIST, COMPLAINT_KEYWORDS
)
# This step:
# - Creates a web interface using Gradio
# - Allows users to input ticket text
# - Returns predictions for issue type, urgency level, and extracted entities
# - Opens a local web server where users can interact with the trained models 