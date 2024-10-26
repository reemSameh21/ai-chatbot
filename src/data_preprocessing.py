import pandas as pd
import spacy
import os

nlp = spacy.load("en_core_web_sm")

def preprocess_text(text: str) -> str:
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if not token.is_stop and token.is_alpha]
    return " ".join(tokens)

def preprocess_data(input_csv: str, output_csv: str):
    columns = ['question', 'answer']
    df = pd.read_csv(input_csv, sep='\t', names=columns)

    df['cleaned_question'] = df['question'].apply(preprocess_text)
    df['cleaned_answer'] = df['answer'].apply(preprocess_text)

    df.to_csv(output_csv, index=False)
    print(f"Preprocessed dataset saved to {output_csv}")


if __name__ == "__main__":
    input_csv = "data/dialogs.txt"
    output_csv = "data/cleaned_data.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    preprocess_data(input_csv, output_csv)