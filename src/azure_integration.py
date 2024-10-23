from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

# Configure Azure Cognitive Services
endpoint = "https://ai1chatbot.cognitiveservices.azure.com/"
credential = AzureKeyCredential("ebe4649649b24b8bb6d9a9088b80bb96")

client = TextAnalyticsClient(endpoint=endpoint, credential=credential)


def analyze_sentiment(user_message: str):
    """Analyze sentiment of a user's message and return the sentiment category."""
    try:
        # Use the TextAnalyticsClient to analyze sentiment
        response = client.analyze_sentiment([user_message])

        # Check if the response is valid and contains results
        if response and len(response) > 0:
            sentiment = response[0].sentiment  # Extract sentiment from the response
            return sentiment
        else:
            return "neutral"  # Default if no sentiment found
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "neutral"  # Fallback response in case of an error


def chatbot_response(user_message: str):
    """Generate a chatbot response based on the sentiment of the user's message."""
    # Step 1: Analyze sentiment
    sentiment = analyze_sentiment(user_message)

    # Step 2: Adjust response based on sentiment
    if sentiment == "negative":
        response = "I'm sorry you're having trouble. Can I help you with something specific?"
    elif sentiment == "positive":
        response = "Great! I'm glad things are going well. How can I assist you today?"
    else:
        response = "Let me know if you need any assistance."

    return response


if __name__ == "__main__":
    user_message = "I'm struggling with the installation!"

    # Generate and print the chatbot response
    response = chatbot_response(user_message)
    print(f"Chatbot Response: {response}")