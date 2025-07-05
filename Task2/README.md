# RAG Quote Retrieval System

This project implements a modular Retrieval-Augmented Generation (RAG) pipeline for intelligent quote retrieval using the [Abirate/english_quotes](https://huggingface.co/datasets/abirate/english_quotes) dataset. It features a fine-tuned sentence embedding model, FAISS vector search, and a user-friendly Streamlit web app.

## Features
- **Data Preparation:** Cleans and processes the quotes dataset.
- **Model Fine-tuning:** Fine-tunes a SentenceTransformer model for semantic search.
- **FAISS Indexing:** Builds a fast vector index for efficient retrieval.
- **RAG Pipeline:** Retrieves relevant quotes and generates answers.
- **Evaluation:** (Optional) Evaluate retrieval quality with RAGAS (placeholder).
- **Modular Streamlit App:** Interactive web interface with clean, modular architecture.

## Project Structure
```
Task2/
  app/                     # Modular Streamlit application
    __init__.py           # Package initialization
    config.py             # Configuration and styling
    data_loader.py        # Component loading and caching
    search_engine.py      # Core search functionality
    ui_components.py      # UI rendering components
    main.py               # Main application orchestration
  app_streamlit.py        # Streamlit entry point
  data_preparation.py     # Data cleaning and preparation
  model_finetuning.py     # Model fine-tuning script
  rag_pipeline.py         # FAISS indexing and retrieval functions
  rag_evaluation.py       # (Optional) Evaluation script
  requirements.txt        # Python dependencies
  pipeline_artifacts/     # Saved model, index, and data artifacts
  improved_quote_retriever/ # Model and tokenizer files
```

## Setup Instructions

1. **Clone the repository and navigate to Task2:**
   ```sh
   cd Task2
   ```

2. **(Recommended) Create and activate a virtual environment:**
   ```sh
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```

4. **Prepare the pipeline artifacts:**
   - Run the scripts in this order if you have not already:
     ```sh
     python data_preparation.py
     python model_finetuning.py
     python rag_pipeline.py
     ```
   - This will create the necessary files in `pipeline_artifacts/` and `improved_quote_retriever/`.

5. **Run the Streamlit app:**
   ```sh
   streamlit run app_streamlit.py
   ```
   - The app will be available at [http://localhost:8501](http://localhost:8501)

## Modular Architecture

The application is now organized into a clean, modular structure:

- **config.py**: Contains all configuration settings, styling, and constants
- **data_loader.py**: Handles loading and caching of models, data, and FAISS index
- **search_engine.py**: Contains the core search functionality and QuoteSearchEngine class
- **ui_components.py**: Manages all UI rendering components and user interactions
- **main.py**: Orchestrates the entire application flow

This modular design makes the code more maintainable, testable, and easier to understand.

## Troubleshooting
- **ModuleNotFoundError:**
  - Ensure you are using the correct Python environment. Activate your venv before installing and running.
  - Install missing packages with `pip install <package-name>`.
- **FAISS or Torch errors:**
  - Make sure you have compatible versions of `faiss-cpu` and `torch` as specified in `requirements.txt`.
- **Artifacts not found:**
  - Ensure you have run all pipeline scripts to generate the required files in `pipeline_artifacts/`.

## Customization
- You can fine-tune the model on your own data by modifying `model_finetuning.py`.
- Adjust retrieval and scoring logic in `rag_pipeline.py` as needed.
- Modify UI components in `app/ui_components.py` for custom styling.
- Update configuration in `app/config.py` for different settings.

## Credits
- Quotes dataset: [Abirate/english_quotes](https://huggingface.co/datasets/abirate/english_quotes)
- Sentence Transformers: https://www.sbert.net/
- FAISS: https://github.com/facebookresearch/faiss

---

For any issues, please open an issue or contact the maintainer. 