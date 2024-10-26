from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential

endpoint = "https://ai1chatbot.cognitiveservices.azure.com/"
credential = AzureKeyCredential("ebe4649649b24b8bb6d9a9088b80bb96")

client = TextAnalyticsClient(endpoint, credential)

def analyze_sentiment(client, message: str):
    documents = [message]
    response = client.analyze_sentiment(documents=documents)[0]
    print(f"Sentiment: {response.sentiment}")
    print(f"Confidence Scores: {response.confidence_scores}")
    return response.sentiment

if __name__ == "__main__":
    while True:
        user_message = input("You: ")

        if user_message.lower() in ["exit", "quit", "stop"]:
            print("Chatbot: Goodbye! Have a great day!")
            break

        sentiment = analyze_sentiment(client, user_message)
        print(f"The sentiment of the message is: {sentiment}")