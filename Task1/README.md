# Customer Support Ticket Classification & Entity Extraction Pipeline

## Overview
This project implements a modular machine learning pipeline to classify customer support tickets by issue type and urgency level, and extract key entities (product names, dates, complaint keywords) from ticket text. It supports both batch and interactive (Gradio web app) usage.

## Project Structure
```
Task1/
  data_preparation.py      # Data loading, cleaning, preprocessing, product/keyword lists
  feature_engineering.py   # Feature extraction (categorical/text features, encoders)
  multi_task_learning.py   # Model training, evaluation, comparison, and selection
  entity_extraction.py     # Entity extraction (products, dates, keywords)
  integration.py           # Integration: prediction + entity extraction for a ticket
  gradio_interface.py      # Gradio web app interface
  app.py                   # Main script: runs the full pipeline and launches Gradio
  requirements.txt         # Python dependencies
  README.md                # This file
  eda_outputs/             # EDA plots and confusion matrices
  models/                  # Saved models and encoders
  model_report.txt         # Model comparison report (auto-generated)
```

## Setup Instructions
1. **Create and activate a virtual environment:**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources:**
   The first run will automatically download required NLTK data (stopwords, punkt).

4. **Place the dataset:**
   Ensure `ai_dev_assignment_tickets_complex_1000.xls` is in the project root or update the path in `app.py`.

## Running the Pipeline
- To train models, generate EDA outputs, and launch the Gradio app, run:
  ```bash
  python app.py
  ```
- The Gradio web app will open in your browser for interactive predictions.
- All EDA plots and confusion matrices are saved in `eda_outputs/`.
- Model comparison results are saved in `model_report.txt`.
- Trained models and encoders are saved in `models/`.

## Module Descriptions
- **data_preparation.py**: Functions for loading, cleaning, and imputing ticket data. Also defines product and complaint keyword lists.
- **feature_engineering.py**: Functions for extracting categorical and text-based features, and fitting encoders.
- **multi_task_learning.py**: Trains and compares multiple models for both issue type and urgency prediction. Selects the best model for each and saves a detailed report.
- **entity_extraction.py**: Functions for extracting product names, dates, and complaint keywords from text.
- **integration.py**: Combines model predictions and entity extraction for a single ticket.
- **gradio_interface.py**: Defines and launches the Gradio web app for interactive use.
- **app.py**: Orchestrates the full pipeline: data prep, feature engineering, model training, and launches the Gradio app.

## Outputs
- **EDA Visualizations:** All plots and confusion matrices are saved in `eda_outputs/`.
- **Model Report:** Model comparison (accuracy, F1, classification report) is saved in `model_report.txt`.
- **Models:** Trained models and encoders are saved in `models/`.
- **Interactive App:** Gradio web app for real-time ticket classification and entity extraction.

## Customization
- Update complaint keywords or product extraction logic in `data_preparation.py` or `entity_extraction.py` as needed.
- You can extend the pipeline with additional features or models by editing the relevant modules.

## Notes
- For questions or issues, please refer to the code comments and docstrings.
- The pipeline is modular and easy to extend for new features or data sources. 