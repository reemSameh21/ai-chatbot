# Intelligent Chatbot with Custom Prompt Engineering

## Overview
This project involves creating an intelligent chatbot using custom prompt engineering and NLP techniques. The chatbot is deployed using Azure Bot Services and includes advanced features such as sentiment analysis and MLOps tracking.

## Project Structure
- `data/`: Stores the conversational dataset.
- `src/`: Contains Python scripts for data preprocessing, chatbot development, prompt engineering, Azure integration, and MLOps tracking.
- `requirements.txt`: Lists all the dependencies required for the project.
- `run.ps1`: A shell script to run the entire project.

## Steps to Run the Project

1. Clone the repository.
2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate

3. Install dependencies:
```bash
pip install -r requirements.txt
```
4. Run the project:
```bash
./run.ps1
```
## Azure Setup
Make sure to replace `YOUR_AZURE_ENDPOINT` and `YOUR_AZURE_KEY` in azure_integration.py and sentiment_analysis.py with your Azure credentials.