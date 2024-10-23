# run.ps1

# Activate the virtual environment
.\venv\Scripts\Activate

python -m spacy download en_core_web_sm

# Run data preprocessing
python .\src\data_preprocessing.py

# Run the chatbot
python .\src\chatbot_training.py

# Run prompt engineering example
python .\src\prompt_engineering.py

# Run Azure integration example
python .\src\azure_integration.py

# Run Sentiment Analysis example
python .\src\sentiment_analysis.py

# Run MLOps logging
python .\src\mlops_tracking.py