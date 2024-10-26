def generate_prompt(question: str) -> str:
    """Generate a custom prompt for the chatbot based on a user's question."""
    prompt = f"Question: {question}\nAnswer:"
    return prompt

if __name__ == "__main__":
    user_question = "What is artificial intelligence?"
    prompt = generate_prompt(user_question)
    print(prompt)