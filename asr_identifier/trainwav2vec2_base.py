import torch
import numpy as np
from datasets import load_dataset, Audio, concatenate_datasets
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift

# List of accents to use
accents = [
    "english", "afrikaans", "swahili", "south-african-english", 
    "xhosa", "isizulu", "kinyarwanda", "igbo", "tswana", "zulu"
]

# Initialize lists to store our training and testing datasets
train_datasets = []
test_datasets = []

# Load each accent subset
for accent in accents:
    try:
        # Handle the special case for South-African-English
        if accent == "south-african-english":
            # Use train set for training since validation is not available
            train_set = load_dataset(
                "intronhealth/afrispeech-200", 
                name=accent, 
                split="train"
            )
            train_datasets.append(train_set)
        else:
            # Use validation set for training
            try:
                valid_set = load_dataset(
                    "intronhealth/afrispeech-200", 
                    name=accent, 
                    split="train"
                )
            except:
                valid_set = load_dataset(
                    "intronhealth/afrispeech-200", 
                    name=accent, 
                    split="validation"
                )
            train_datasets.append(valid_set)
        
        # Load test set
        test_set = load_dataset(
            "intronhealth/afrispeech-200", 
            name=accent, 
            split="test"
        )
        test_datasets.append(test_set)
        
        print(f"Successfully loaded {accent} dataset")
    except Exception as e:
        print(f"Error loading {accent} dataset: {e}")


# Create a label mapping
label2id = {accent: i for i, accent in enumerate(accents)}
id2label = {i: accent for i, accent in enumerate(accents)}

# Function to add accent labels to datasets
def add_accent_label(example, accent):
    example["label"] = label2id[accent]
    return example

# Add labels to each dataset
for i, accent in enumerate(accents):
    if i < len(train_datasets):
        train_datasets[i] = train_datasets[i].map(
            lambda example: add_accent_label(example, accent)
        )
    
    if i < len(test_datasets):
        test_datasets[i] = test_datasets[i].map(
            lambda example: add_accent_label(example, accent)
        )

# Combine all training datasets
combined_train_dataset = concatenate_datasets(train_datasets)

# Combine all test datasets
combined_test_dataset = concatenate_datasets(test_datasets)

print(f"Combined training dataset size: {len(combined_train_dataset)}")
print(f"Combined test dataset size: {len(combined_test_dataset)}")


# Load the feature extractor for wav2vec2
feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

# Ensure all the necessary columns are present
combined_train_dataset = combined_train_dataset.cast_column("audio", Audio(sampling_rate=16000))
combined_test_dataset = combined_test_dataset.cast_column("audio", Audio(sampling_rate=16000))

# Create audio data augmentation pipeline
audio_augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# Prepare the datasets with augmentation for training
def prepare_train_dataset(examples):
    audio = examples["audio"]
    waveform = audio["array"]
    sample_rate = audio["sampling_rate"]
    
    # Apply augmentation with 50% probability
    if np.random.random() < 0.5:
        waveform = audio_augmenter(samples=waveform, sample_rate=sample_rate)
    
    features = feature_extractor(
        waveform, 
        sampling_rate=sample_rate,
        max_length=16000*5,  # limit to 5 seconds
        truncation=True,
        padding="max_length",
    )
    
    examples["input_values"] = features.input_values[0]
    return examples

# Prepare the datasets without augmentation for testing
def prepare_test_dataset(examples):
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

# Apply preprocessing to both datasets
processed_train_dataset = combined_train_dataset.map(
    prepare_train_dataset, 
    remove_columns=[col for col in combined_train_dataset.column_names if col != "label"],
    num_proc=4  # Adjust based on your CPU cores
)

processed_test_dataset = combined_test_dataset.map(
    prepare_test_dataset, 
    remove_columns=[col for col in combined_test_dataset.column_names if col != "label"],
    num_proc=4  # Adjust based on your CPU cores
)

# Load a pre-trained wav2vec2 model and adapt it for classification
num_labels = len(accents)
model = AutoModelForAudioClassification.from_pretrained(
    "facebook/wav2vec2-base", 
    num_labels=num_labels,
    label2id=label2id,
    id2label=id2label
)

# Define metrics for evaluation
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted")
    }

# Define training arguments
training_args = TrainingArguments(
    output_dir="./accent-identifier",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    push_to_hub=False,
    logging_dir="./logs",
    logging_steps=10,
    gradient_accumulation_steps=2,  # Helps with memory usage
    fp16=True,  # Use mixed precision training if available
)

# Create the trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=processed_train_dataset,
    eval_dataset=processed_test_dataset,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model
model_path = "./accent-identifier-final"
trainer.save_model(model_path)
feature_extractor.save_pretrained(model_path)


# Evaluate the model on the test set
test_results = trainer.evaluate(processed_test_dataset)
print(f"Test results: {test_results}")

# Create a confusion matrix to see how well the model distinguishes between accents
predictions = trainer.predict(processed_test_dataset)
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

cm = confusion_matrix(labels, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=accents)
plt.figure(figsize=(15, 15))
disp.plot(cmap=plt.cm.Blues)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.close()
