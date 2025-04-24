import torch
import numpy as np
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift


def predict_accent(audio_path, model_path="./accent-identifier-final"):
    """
    Predict the accent of a given audio file.
    
    Args:
        audio_path (str): Path to the audio file
        model_path (str): Path to the saved model
        
    Returns:
        dict: Prediction results including predicted accent and confidence scores
    """
    from datasets import Dataset, Audio
    
    # Load model and feature extractor
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    
    # Load audio
    dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
    
    # Preprocess audio
    def preprocess(examples):
        audio = examples["audio"]
        features = feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            max_length=16000*5,  # limit to 5 seconds
            truncation=True,
            padding="max_length",
        )
        examples["input_values"] = features.input_values[0]
        return examples
    
    processed = dataset.map(preprocess, remove_columns=dataset.column_names)
    
    # Make prediction
    input_values = torch.tensor(processed["input_values"], dtype=torch.float32)
    with torch.no_grad():
        outputs = model(input_values)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class_id = torch.argmax(predictions, dim=-1).item()
        predicted_accent = model.config.id2label[predicted_class_id]
    
    # Get confidence scores for all accents
    confidence_scores = {model.config.id2label[i]: score.item() for i, score in enumerate(predictions[0])}
    
    return {
        "predicted_accent": predicted_accent,
        "confidence": predictions[0][predicted_class_id].item(),
        "all_scores": confidence_scores
    }

# Example of how to use the inference function
def inference_example():
    # Replace with actual audio path
    audio_file = "path/to/your/audio/file.wav"
    result = predict_accent(audio_file)
    
    print(f"Predicted accent: {result['predicted_accent']}")
    print(f"Confidence: {result['confidence']:.2f}")
    print("\nAll accent scores:")
    
    # Sort and display all scores
    sorted_scores = sorted(
        result['all_scores'].items(), 
        key=lambda x: x[1], 
        reverse=True
    )
    
    for accent, score in sorted_scores:
        print(f"{accent}: {score:.4f}")

# If you want to run the inference example
inference_example()
