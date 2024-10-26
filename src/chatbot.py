import os
from azure.identity import DefaultAzureCredential
from azure.ai.openai import OpenAIClient

# Configure Azure OpenAI
endpoint = "https://YOUR_AZURE_OPENAI_ENDPOINT"  # Replace with your Azure OpenAI endpoint
credential = DefaultAzureCredential()  # This will use the default Azure credential

client = OpenAIClient(endpoint=endpoint, credential=credential)


def generate_response(user_message: str):
    """Generate a response from Azure OpenAI based on user input."""
    try:
        # Generate a response using Azure OpenAI's ChatGPT model
        response = client.chat.completions.create(
            deployment_id="YOUR_MODEL_DEPLOYMENT_ID",  # Replace with your model deployment ID
            messages=[
                {"role": "user", "content": user_message}
            ]
        )

        # Extract the chatbot's reply
        return response.choices[0].message['content']
    except Exception as e:
        print(f"Error generating response: {e}")
        return "I'm having trouble generating a response right now."


def main():
    print("Welcome to the Chatbot! Type 'exit' to end the conversation.")

    while True:
        user_message = input("You: ")
        if user_message.lower() == 'exit':
            print("Chatbot: Goodbye!")
            break

        # Generate and print the chatbot response
        response = generate_response(user_message)
        print(f"Chatbot: {response}")


if __name__ == "__main__":
    main()