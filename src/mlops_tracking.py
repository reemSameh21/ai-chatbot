import mlflow
import mlflow.pytorch
from transformers import GPT2LMHeadModel

mlflow.set_tracking_uri("http://localhost:5000")

def train_and_log_model():
    """Train the model and log the parameters and metrics in MLflow."""
    with mlflow.start_run():
        # Initialize the GPT model
        model = GPT2LMHeadModel.from_pretrained("gpt2")
        mlflow.log_param("model_name", "gpt2")

        # Simulate training and log some metrics
        mlflow.log_metric("training_loss", 0.123)
        mlflow.pytorch.log_model(model, "chatbot_model")
        print("Model logged successfully!")

if __name__ == "__main__":
    train_and_log_model()