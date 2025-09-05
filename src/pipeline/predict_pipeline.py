import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import numpy as np

class PredictPipeline:
    def __init__(self, model_type: str = "full_ft"):
        """
        Args:
            model_type (str): Which model to use: "full_ft", "partial_ft", or "lora"
        """
        artifact_mapping = {
            "full_ft": os.path.join("artifacts", "full_ft_model_distilBERT"),
            "partial_ft": os.path.join("artifacts", "partial_ft_model"),
            "lora": os.path.join("artifacts", "lora_modeldistilBERT"),
        }

        if model_type not in artifact_mapping:
            raise ValueError(f"Invalid model_type. Must be one of {list(artifact_mapping.keys())}")

        self.model_path = artifact_mapping[model_type]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()

        # Label mapping
        self.id2label = {0: "normal", 1: "offensive", 2: "hatespeech"}

    def predict(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Predict labels and probabilities for a list of texts.

        Args:
            texts (List[str]): List of input comments

        Returns:
            List[Dict[str, Any]]: List of dictionaries with prediction and probabilities
        """
        # Tokenize inputs
        inputs = self.tokenizer(
            texts, 
            padding=True, 
            truncation=True, 
            max_length=128, 
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Model inference
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=-1)

        # Map predictions and probabilities to labels
        results = []
        for text, pred, prob in zip(texts, preds, probs):
            results.append({
                "text": text,
                "predicted_label": self.id2label[pred],
                "probabilities": prob.tolist()  # convert to list for JSON serialization
            })

        return results



if __name__ == "__main__":
    # Example usage
    pipeline = PredictPipeline(model_type="full_ft")

    sample_texts = [
        "i love your hair",
        "What a lovely day!",
        "fuck you"
    ]

    predictions = pipeline.predict(sample_texts)
    for p in predictions:
        print(p)
