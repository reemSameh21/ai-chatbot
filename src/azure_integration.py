from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai1chatbot.cognitiveservices.azure.com/"
credential = AzureKeyCredential("ebe4649649b24b8bb6d9a9088b80bb96")

client = TextAnalyticsClient(endpoint=endpoint, credential=credential)

def analyze_sentiment(user_message: str):
    """Analyze sentiment of a user's message and return the sentiment category."""
    try:
        response = client.analyze_sentiment([user_message])

        if response and len(response) > 0:
            sentiment = response[0].sentiment
            return sentiment
        else:
            return "neutral"
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return "neutral"

def chatbot_response(user_message: str):
    """Generate a chatbot response based on the sentiment of the user's message."""
    # Step 1: Analyze sentiment
    sentiment = analyze_sentiment(user_message)

    # Step 2: Adjust response based on sentiment
    if sentiment == "negative":
        response = "I'm sorry you're having trouble."
    elif sentiment == "positive":
        response = "Great! I'm glad things are going well."
    else:
        response = "Let me know if you need any assistance."

    return response

if __name__ == "__main__":
    while True:
        user_message = input("You: ")

        if user_message.lower() in ["exit", "quit", "stop"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        response = chatbot_response(user_message)
        print(f"Chatbot: {response}")