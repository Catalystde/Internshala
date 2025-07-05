# Multi-Task Learning Module for Support Ticket Classification
# This module trains and compares multiple machine learning models for two tasks:
# 1. Predicting the issue type (e.g., Product Defect, Billing Problem)
# 2. Predicting the urgency level (e.g., Low, Medium, High)

import pandas as pd  # For data manipulation
import numpy as np   # For numerical operations
from sklearn.model_selection import train_test_split, cross_val_score  # For splitting data and cross-validation
from sklearn.linear_model import LogisticRegression  # Linear classification model
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier  # Ensemble models
from sklearn.svm import SVC  # Support Vector Machine classifier
from sklearn.neighbors import KNeighborsClassifier  # K-Nearest Neighbors classifier
from sklearn.naive_bayes import MultinomialNB  # Naive Bayes classifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report  # For model evaluation
import matplotlib.pyplot as plt  # For creating plots
import seaborn as sns  # For enhanced plotting
import os  # For file and directory operations
import warnings  # For suppressing warnings
warnings.filterwarnings('ignore')  # Suppress all warnings for cleaner output

def train_and_select_models(df, le_issue, le_product, le_failure, le_delay, le_urgency, eda_dir='eda_outputs'):
    """
    Train multiple models for both issue type and urgency level prediction, then select the best ones.
    
    Args:
        df (pd.DataFrame): DataFrame with all features and target variables
        le_issue, le_product, le_failure, le_delay, le_urgency: LabelEncoder objects
        eda_dir (str): Directory to save exploratory data analysis outputs
        
    Returns:
        tuple: (best_issue_model, best_urgency_model, features, X, le_urgency, le_issue, le_product, le_failure, le_delay)
    """
    # Define the features (input variables) for our models
    # These are the engineered features we created in feature_engineering.py
    features = ['issue_type_encoded', 'product_encoded', 'high_urgency_keywords',
                'support_contact', 'failure_time_encoded', 'delay_duration_encoded']
    
    # Prepare the input features (X) and target variables (y)
    X = df[features]  # All input features
    y_urgency = df['urgency_level']  # Target for urgency prediction
    y_issue = df['issue_type']       # Target for issue type prediction
    
    # Encode the target variables to numerical values for model training
    y_encoded = le_urgency.fit_transform(y_urgency)      # Convert urgency levels to numbers
    y_issue_encoded = le_issue.fit_transform(y_issue)    # Convert issue types to numbers
    
    # Split the data into training and testing sets for urgency prediction
    # test_size=0.2 means 80% for training, 20% for testing
    # stratify ensures the split maintains the same proportion of classes
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # Split the data into training and testing sets for issue type prediction
    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(X, y_issue_encoded, test_size=0.2, random_state=42, stratify=y_issue_encoded)
    
    # Define all the models we want to compare
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),  # Linear model with increased iterations
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),  # Ensemble of decision trees
        'SVM': SVC(probability=True),  # Support Vector Machine with probability estimates
        'KNN': KNeighborsClassifier(),  # K-Nearest Neighbors
        'Gradient Boosting': GradientBoostingClassifier(),  # Boosting ensemble
        'Naive Bayes': MultinomialNB()  # Probabilistic classifier
    }
    
    report_lines = []  # List to store all evaluation results
    
    # ===== MODEL SELECTION FOR ISSUE TYPE PREDICTION =====
    best_issue_model = None
    best_issue_acc = 0
    report_lines.append('Model Comparison for issue_type prediction:\n')
    
    # Test each model for issue type prediction
    for name, model in models.items():
        if name == 'Naive Bayes':
            # Naive Bayes requires non-negative features, so we handle negative values
            X_train_nb = X_train_i.copy()
            X_test_nb = X_test_i.copy()
            X_train_nb[X_train_nb < 0] = 0  
            X_test_nb[X_test_nb < 0] = 0
            model.fit(X_train_nb, y_train_i)
            y_pred = model.predict(X_test_nb)
        else:
            # Train the model and make predictions
            model.fit(X_train_i, y_train_i)
            y_pred = model.predict(X_test_i)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test_i, y_pred)  # Overall accuracy
        f1 = f1_score(y_test_i, y_pred, average='weighted', zero_division=0)  # F1 score (harmonic mean of precision and recall)
        
        # Add results to report
        report_lines.append(f'{name} - issue_type\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n')
        report_lines.append(classification_report(y_test_i, y_pred, target_names=le_issue.classes_, zero_division=0))
        report_lines.append('\n')
        
        # Keep track of the best performing model
        if acc > best_issue_acc:
            best_issue_acc = acc
            best_issue_model = model
    
    # ===== MODEL SELECTION FOR URGENCY LEVEL PREDICTION =====
    best_urgency_model = None
    best_urgency_acc = 0
    report_lines.append('Model Comparison for urgency_level prediction:\n')
    
    # Test each model for urgency level prediction
    for name, model in models.items():
        if name == 'Naive Bayes':
            # Handle negative values for Naive Bayes
            X_train_nb = X_train.copy()
            X_test_nb = X_test.copy()
            X_train_nb[X_train_nb < 0] = 0
            X_test_nb[X_test_nb < 0] = 0
            model.fit(X_train_nb, y_train)
            y_pred = model.predict(X_test_nb)
        else:
            # Train the model and make predictions
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        
        # Calculate performance metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        # Add results to report
        report_lines.append(f'{name} - urgency_level\nAccuracy: {acc:.4f}\nF1 Score: {f1:.4f}\n')
        report_lines.append(classification_report(y_test, y_pred, target_names=le_urgency.classes_, zero_division=0))
        report_lines.append('\n')
        
        # Keep track of the best performing model
        if acc > best_urgency_acc:
            best_urgency_acc = acc
            best_urgency_model = model
    
    # Save the detailed model comparison report to a file
    with open('model_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    # Return the best models and all necessary components for later use
    return best_issue_model, best_urgency_model, features, X, le_urgency, le_issue, le_product, le_failure, le_delay 