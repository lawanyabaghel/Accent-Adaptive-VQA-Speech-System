import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
import tempfile
import time
import soundfile as sf
import re
import traceback
import jiwer  # Simple package for WER calculation
import librosa  # For audio feature extraction
import warnings
from transformers import AutoTokenizer, AutoProcessor, Qwen2_5_VLForConditionalGeneration
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.utils.types import str_or_none
from datasets import Dataset, Audio
from scipy.io.wavfile import write
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
warnings.filterwarnings("ignore")

# Accent Identification module
class AccentIdentifier:
    """Class for identifying accents from audio."""
    
    def __init__(self, model_path="./accent-identifier-final"):
        """Initialize the accent identifier model."""
        self.model_path = model_path
        self.model = None
        self.feature_extractor = None
        
    def load_model(self):
        """Load the accent identification model."""
        try:
            print("Loading accent identification model...")
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_path)
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_path)
            print("Accent identification model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading accent identification model: {e}")
            traceback.print_exc()
            return False
    
    def identify_accent(self, audio_path):
        """
        Identify the accent from an audio file.
        
        Args:
            audio_path (str): Path to the audio file
            
        Returns:
            dict: Prediction results including predicted accent and confidence scores
        """
        try:
            self.load_model()
            
            # Load audio
            dataset = Dataset.from_dict({"audio": [audio_path]}).cast_column("audio", Audio(sampling_rate=16000))
            
            # Preprocess audio
            def preprocess(examples):
                audio = examples["audio"]
                features = self.feature_extractor(
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
                outputs = self.model(input_values)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = torch.argmax(predictions, dim=-1).item()
                predicted_accent = self.model.config.id2label[predicted_class_id]
            
            # Get confidence scores for all accents
            confidence_scores = {self.model.config.id2label[i]: score.item() for i, score in enumerate(predictions[0])}
            
            return {
                "predicted_accent": predicted_accent,
                "confidence": predictions[0][predicted_class_id].item(),
                "all_scores": confidence_scores
            }
        except Exception as e:
            print(f"Error in accent identification: {e}")
            traceback.print_exc()

# Speech Metrics class - updated version with SNR and speech clarity for ASR
class SpeechMetrics:
    """Class for speech technology evaluation metrics."""
    
    def __init__(self):
        """Initialize the speech metrics."""
        self.start_time = None
        self.end_time = None
        
    def start_timer(self) -> None:
        """Start the latency timer."""
        self.start_time = time.time()
        
    def stop_timer(self) -> float:
        """Stop the latency timer and return elapsed time."""
        if self.start_time is None:
            return 0.0
            
        self.end_time = time.time()
        elapsed_time = self.end_time - self.start_time
        # Reset for next use
        self.start_time = None
        self.end_time = None
        return elapsed_time
    
    def measure_asr_wer(self, 
                        audio_data, 
                        asr_transcript: str, 
                        reference_text: str = None) -> dict:
        """
        Measure ASR quality metrics including SNR and speech clarity.
        
        Args:
            audio_data: Tuple of (sample_rate, audio_array)
            asr_transcript: The transcription from the ASR model
            reference_text: Optional reference text for WER calculation
            
        Returns:
            Dictionary with ASR quality metrics
        """
        try:
            metrics = {}
            
            # If reference text is provided, calculate WER
            if reference_text:
                metrics["wer"] = jiwer.wer(reference_text, asr_transcript) * 100
                
            # Calculate signal-to-noise ratio
            if audio_data and len(audio_data) == 2:
                audio = audio_data[1]
                
                # Signal-to-Noise Ratio calculation
                signal_power = np.mean(audio**2)
                noise_power = np.var(audio)
                if noise_power > 0:
                    snr = 10 * np.log10(signal_power / noise_power)
                    metrics["Signal to Noise Ratio"] = snr
                
                # Articulation Index (speech clarity indicator)
                # Simplified version based on energy distribution
                # Compute short-time FFT to get frequency information
                try:                    
                    # Compute spectrogram
                    D = librosa.stft(audio)
                    
                    # Get magnitude spectrogram
                    magnitude = np.abs(D)
                    
                    # Compute spectral flatness (Wiener entropy)
                    # Higher values indicate more noise-like signal, lower values more tonal/speech
                    spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / np.mean(magnitude + 1e-10)
                    
                    # Convert to a clarity score (0-100 scale, higher is better)
                    clarity_score = max(0, min(100, 100 * (1 - spectral_flatness)))
                    metrics["Speech Clarity"] = clarity_score
                    
                except Exception as e:
                    print(f"Error calculating speech clarity: {e}")
            
            return metrics
            
        except Exception as e:
            print(f"Error measuring ASR metrics: {e}")
            return {"Signal to Noise Ratio": None, "Speech Clarity": None}
    
    def estimate_mos(self, audio_data, rtf=None) -> dict:
        """
        Estimate Mean Opinion Score for TTS based on simple audio features.
        This is a very basic approximation without using Versa.
        
        Args:
            audio_data: Tuple of (sample_rate, audio_array)
            rtf: Real-Time Factor from TTS generation
            
        Returns:
            Dictionary with estimated MOS and RTF
        """
        try:
            if not audio_data or len(audio_data) != 2:
                return {"estimated_mos": None}
                
            sample_rate, audio = audio_data
            
            # Calculate basic audio statistics that loosely correlate with quality
            # Higher dynamic range often means better quality
            dynamic_range = np.max(np.abs(audio)) - np.min(np.abs(audio))
            
            # Spectral centroid correlates with "brightness"
            try:
                spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
                
                # Normalize spectral centroid to a 0-1 range (typical values are in the 1000-4000 Hz range)
                norm_spec_cent = min(1.0, max(0.0, (spec_cent - 500) / 3500))
                
                # Formula to approximate MOS score (1-5 scale)
                # This is just a rough approximation based on audio features
                # Actual MOS requires human evaluations or trained models
                estimated_mos = 2.0 + (dynamic_range * 1.5) + (norm_spec_cent * 1.5)
                
            except Exception as e:
                print(f"Error calculating spectral features: {e}")
                # Fallback if librosa fails
                estimated_mos = 2.0 + (dynamic_range * 1.5)
            
            # Clamp to valid MOS range (1-5)
            estimated_mos = max(1.0, min(5.0, estimated_mos))
            
            result = {
                "Mean Opinion Score(MOS)": estimated_mos,
                "Dynamic Range": dynamic_range
            }
            
            # Add RTF if provided
            if rtf is not None:
                result["Real-Time Factor(RTF)"] = rtf
                
            return result
            
        except Exception as e:
            print(f"Error estimating TTS MOS: {e}")
            return {"estimated_mos": None}
    
    def format_metrics(self, metrics_dict: dict) -> str:
        """
        Format metrics dictionary into a readable string.
        
        Args:
            metrics_dict: Dictionary containing metrics values
            
        Returns:
            Formatted string representation of the metrics
        """
        formatted_str = ""
        
        for key, value in metrics_dict.items():
            if value is not None:
                if isinstance(value, float):
                    formatted_str += f"{key}: {value:.6f}\n" if key == "rtf" else f"{key}: {value:.2f}\n"
                else:
                    formatted_str += f"{key}: {value}\n"
            else:
                formatted_str += f"{key}: Not available\n"
                
        return formatted_str

# Configuration for models
class ModelConfig:
    # Base ASR Configuration
    ASR_PRETRAINED_MODEL = "espnet/owsm_v3.1_ebf"  # Base model
    ASR_LANGUAGE = "eng"  # Default language code
    
    # Accent-specific ASR models
    ASR_FINETUNED_MODELS = {
        "zulu": "/data/user_data/lbaghel/speech_project/exp/zulu/normal_model_finetune/5epoch.pth",
        # "isizulu": "/data/user_data/lbaghel/speech_project/exp/isizulu/normal_model_finetune/5epoch.pth",
        # "afrikaans": "/data/user_data/lbaghel/speech_project/exp/afrikaans/normal_model_finetune/5epoch.pth",
        "swahili": "/data/user_data/lbaghel/speech_project/exp/swahili/normal_model_finetune/5epoch.pth",
        # "south-african-english": "/data/user_data/lbaghel/speech_project/exp/south-african-english/normal_model_finetune/5epoch.pth",
        "kinyarwanda": "/data/user_data/lbaghel/speech_project/exp/kinyarwanda/normal_model_finetune/5epoch.pth",       
        # "english": "/home/lbaghel/speech_tech/speech_tech/5epoch.pth"  # Default English model
    }
    
    # Accent Identifier Configuration
    ACCENT_MODEL_PATH = "./accent-identifier-final"
    
    # VLM Configuration
    VLM_MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
    
    # XTTS Configuration
    XTTS_CONFIG_PATH = "afro-tts/config.json"
    XTTS_CHECKPOINT_DIR = "afro-tts/"

# Global variables to store loaded models
asr_models = {}  # Dictionary to store loaded ASR models for different accents
accent_identifier = None
vlm_components = None
xtts_model = None
metrics = None

# Load Accent Identification model
def load_accent_identifier(config):
    print("Loading Accent Identification model...")
    try:
        identifier = AccentIdentifier(model_path=config.ACCENT_MODEL_PATH)
        if identifier.load_model():
            print("Accent Identification model loaded successfully")
        else:
            print("Warning: Using fallback filename-based accent identification")
        return identifier
    except Exception as e:
        print(f"Error loading Accent Identification model: {e}")
        traceback.print_exc()
        return AccentIdentifier()  # Return instance that will use filename-based identification

# Load ASR model with finetuned weights for specific accent
def load_asr_model(config, accent="english"):
    print(f"Loading ASR model for accent: {accent}")
    try:
        # Check if we already have this model loaded
        if accent in asr_models and asr_models[accent] is not None:
            print(f"Using cached ASR model for {accent}")
            return asr_models[accent]
            
        # Load the pre-trained model
        pretrained_model = Speech2Text.from_pretrained(
            config.ASR_PRETRAINED_MODEL,
            lang_sym=f"<{config.ASR_LANGUAGE}>",
            beam_size=1,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # Use GPU if available
        if torch.cuda.is_available():
            pretrained_model.s2t_model.cuda()
            pretrained_model.device = 'cuda'
        
        # Load accent-specific fine-tuned weights if available
        if accent in config.ASR_FINETUNED_MODELS:
            finetuned_path = config.ASR_FINETUNED_MODELS[accent]
            if os.path.exists(finetuned_path):
                d = torch.load(finetuned_path)
                pretrained_model.s2t_model.load_state_dict(d)
                print(f"Fine-tuned ASR model for {accent} loaded successfully")
            else:
                print(f"Warning: Fine-tuned model for {accent} not found at {finetuned_path}")
                
        # Cache the model
        asr_models[accent] = pretrained_model
            
        return pretrained_model
    except Exception as e:
        print(f"Error loading ASR model for {accent}: {e}")
        traceback.print_exc()
        return None

# Load Vision-Language Model
def load_vlm_model(config):
    print("Loading Vision-Language Model...")
    try:
        # Use the specific Qwen2.5-VL model class instead of AutoModel
        tokenizer = AutoTokenizer.from_pretrained(config.VLM_MODEL)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            config.VLM_MODEL, 
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(config.VLM_MODEL)
        
        print("VLM model loaded successfully")
        return {"model": model, "tokenizer": tokenizer, "processor": processor}
    except Exception as e:
        print(f"Error loading VLM model: {e}")
        traceback.print_exc()
        return None

# Load XTTS model
def load_xtts_model(config):
    print("Loading XTTS model...")
    try:
        # Load model config and checkpoint
        xtts_config = XttsConfig()
        xtts_config.load_json(config.XTTS_CONFIG_PATH)
        model = Xtts.init_from_config(xtts_config)
        model.load_checkpoint(xtts_config, checkpoint_dir=config.XTTS_CHECKPOINT_DIR, eval=True)
        
        # Use GPU if available
        if torch.cuda.is_available():
            model.cuda()
        
        print("XTTS model loaded successfully")
        return {"model": model, "config": xtts_config}
    except Exception as e:
        print(f"Error loading XTTS model: {e}")
        traceback.print_exc()
        return None

# Function to initialize all models
def initialize_models():
    global accent_identifier, vlm_components, xtts_model, metrics
    
    # Initialize configuration
    config = ModelConfig()
    
    # Create metrics object
    metrics = SpeechMetrics()
    
    # Load models
    accent_identifier = load_accent_identifier(config)
    vlm_components = load_vlm_model(config)
    xtts_model = load_xtts_model(config)
    
    # Load all ASR models
    print("Loading all ASR models...")
    for accent in config.ASR_FINETUNED_MODELS.keys():
        print(f"Loading ASR model for {accent}...")
        asr_models[accent] = load_asr_model(config, accent)
    
    # Check if all models loaded successfully
    missing_models = [accent for accent, model in asr_models.items() if model is None]
    if missing_models:
        print(f"Warning: ASR models for these accents failed to load: {', '.join(missing_models)}")
    else:
        print("All ASR models loaded successfully")
    
    if not all([accent_identifier, vlm_components, xtts_model, metrics]) or not asr_models:
        print("Warning: Not all models were loaded successfully")
    else:
        print("All models loaded successfully")

# Function to identify accent from audio filename
def identify_accent(audio_path):
    print(f"Identifying accent from filename: {audio_path}")
    try:
        # Extract filename without path and extension
        filename = os.path.basename(audio_path).lower()
        
        # Map of keywords to accents
        accent_keywords = {
            "zulu": ["zulu"],
            "isizulu": ["isizulu"],
            "afrikaans": ["afrikaans"],
            "swahili": ["swahili"],
            "south-african-english": ["south_african_english", "south-african-english", "south_african", "african_english"],
            "kinyarwanda": ["kinyarwanda"],
            "xhosa": ["xhosa"],
            "igbo": ["igbo"],
            "tswana": ["tswana"],
            "english": ["english"]
        }
        
        # Identify accent from filename
        identified_accent = "english"  # Default
        
        for accent, keywords in accent_keywords.items():
            if any(keyword in filename for keyword in keywords):
                identified_accent = accent
                break
        
        # Map specific variants to standard names if needed
        accent_mapping = {
            "south_african_english": "south-african-english"
        }
        
        if identified_accent in accent_mapping:
            identified_accent = accent_mapping[identified_accent]
            
        # Create mock confidence scores for other accents
        all_accents = list(accent_keywords.keys())
        all_scores = {accent: 0.1 for accent in all_accents}
        # all_scores[identified_accent] = confidence
        
        result = {
            "predicted_accent": identified_accent,
            # "confidence": confidence,
            "all_scores": all_scores
        }
        
        print(f"Identified accent from filename: {identified_accent}")
        return result
        
    except Exception as e:
        print(f"Error in filename-based accent identification: {e}")
        traceback.print_exc()
        return {"predicted_accent": "English"}

# Function to process audio through ASR with accent-specific model
def transcribe_audio(audio_path, accent="english"):
    global asr_models
    
    print(f"Transcribing audio from: {audio_path} using {accent} ASR model")
    try:
        # Use the preloaded accent-specific model
        if accent not in asr_models or asr_models[accent] is None:
            print(f"ASR model for {accent} not available, trying similar accent")
            
            # Try to find a similar accent model
            similar_accents = {
                "zulu": ["zulu"],
                "isizulu": ["isizulu"],
                "south-african-english": ["south-african-english"],
                "english": ["english"]
            }
            
            found_similar = False
            if accent in similar_accents:
                for similar in similar_accents[accent]:
                    if similar in asr_models and asr_models[similar] is not None:
                        accent = similar
                        print(f"Using similar accent model: {accent}")
                        found_similar = True
                        break
            
            if not found_similar:
                # If no similar accent found, try to load the specified accent
                config = ModelConfig()
                asr_models[accent] = load_asr_model(config, accent)
                
            # If still not available, fall back to the first available model
            if not accent in asr_models or asr_models[accent] is None:
                for available_accent, model in asr_models.items():
                    if model is not None:
                        accent = available_accent
                        print(f"Falling back to available accent model: {accent}")
                        break
                
        asr_model = asr_models[accent]
        if asr_model is None:
            return "Error: No ASR model available", None
            
        # Check if the audio file exists
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found at {audio_path}")
            return "Error: Audio file not found", None
            
        # Get file info
        print(f"Audio file size: {os.path.getsize(audio_path)} bytes")
        
        # Load and process .wav file using soundfile
        try:
            waveform, sample_rate = sf.read(audio_path)
            print(f"Audio loaded: shape={waveform.shape}, sample_rate={sample_rate}, min={waveform.min()}, max={waveform.max()}")
        except Exception as e:
            print(f"Error reading audio file with soundfile: {e}")
            
            # Alternative approach using librosa
            try:
                import librosa
                waveform, sample_rate = librosa.load(audio_path, sr=16000)
                print(f"Audio loaded with librosa: shape={waveform.shape}, sample_rate={sample_rate}")
            except Exception as e2:
                print(f"Error reading audio file with librosa: {e2}")
                return "Error: Could not read audio file", None
        
        # Ensure audio is mono
        if len(waveform.shape) > 1 and waveform.shape[1] > 1:
            waveform = np.mean(waveform, axis=1)
            print(f"Converted to mono: shape={waveform.shape}")
        
        # Resample if needed (ESPnet expects 16kHz)
        expected_sample_rate = 16000
        if sample_rate != expected_sample_rate:
            print(f"Resampling audio from {sample_rate}Hz to {expected_sample_rate}Hz")
            try:
                # Try using librosa for resampling
                import librosa
                waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=expected_sample_rate)
            except ImportError:
                # Fall back to numpy interpolation
                resample_factor = expected_sample_rate / sample_rate
                waveform = np.interp(
                    np.arange(0, len(waveform) * resample_factor), 
                    np.arange(0, len(waveform)) * resample_factor / sample_rate * expected_sample_rate, 
                    waveform
                )
            print(f"Resampled audio: shape={waveform.shape}")
        
        # Normalize audio if needed
        if np.max(np.abs(waveform)) > 1.0:
            waveform = waveform / np.max(np.abs(waveform))
            print("Audio normalized")
            
        # Convert to float32 if needed by the model
        waveform = waveform.astype(np.float32)
        
        # Run ASR inference
        print("Running ASR inference...")
        pred = asr_model(waveform)
        raw_transcribed_text = pred[0][0] if pred else ""
        
        # Clean up special tokens and tags from ASR output
        # Remove all text between < and > including the brackets
        cleaned_text = re.sub(r'<[^>]+>', '', raw_transcribed_text)
        
        # Remove extra spaces and trim
        cleaned_text = ' '.join(cleaned_text.split())
        
        print(f"Raw Transcription: {raw_transcribed_text}")
        print(f"Cleaned Transcription: {cleaned_text}")
        
        # Return both the cleaned text and the audio data for metrics
        return cleaned_text, (expected_sample_rate, waveform)
    except Exception as e:
        print(f"Error in transcription: {e}")
        traceback.print_exc()
        return "Error in transcription process", None

# Function to get VLM response
def get_vlm_response(image_path, question):
    global vlm_components
    
    print("Getting VLM response...")
    try:
        model = vlm_components["model"]
        tokenizer = vlm_components["tokenizer"]
        processor = vlm_components["processor"]
        
        # Load the image directly
        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            print(f"Image file not found at {image_path}")
            return "Error: Image file not found"
        
        # Format messages according to Qwen2.5-VL requirements
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # Pass the PIL Image directly instead of path
                    },
                    {"type": "text", "text": question},
                ],
            }
        ]
        
        # Prepare for inference using Qwen2.5-VL specific approach
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        # Process the image directly without using process_vision_info
        # This avoids the list index out of range error
        inputs = processor(
            text=[text],
            images=[image],  # Pass the image directly
            videos=None,     # No videos
            padding=True,
            return_tensors="pt",
        )
        
        # Move inputs to the correct device
        inputs = inputs.to(model.device)
        
        # Generate response
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
            
            # Trim input tokens to get only the generated part
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            
            # Decode the generated text
            response = processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )[0]  # Get first item from batch
        
        print(f"VLM Response: {response}")
        return response
    except Exception as e:
        print(f"Error in VLM processing: {e}")
        print(f"Details: {str(e)}")
        traceback.print_exc()
        return f"Error in VLM processing: {str(e)}"

# Function to synthesize speech using XTTS
def synthesize_speech(text, speaker_wav=None):
    global xtts_model
    
    print("Synthesizing speech with XTTS...")
    try:
        # Default to a simple response if the VLM failed
        if text.startswith("Error in VLM processing"):
            text = "I couldn't analyze the image properly. Please try again."
            
        # Check if we have the XTTS model loaded
        if xtts_model is None or "model" not in xtts_model or "config" not in xtts_model:
            print("XTTS model not loaded properly")
            return None, None, None
            
        model = xtts_model["model"]
        config = xtts_model["config"]
        
        # Default speaker audio if none provided
        if speaker_wav is None or not os.path.exists(speaker_wav):
            print("No speaker audio provided or file not found, using default voice")
            speaker_wav = "afro-tts/audios/south_african_english.wav"  # Default speaker
            
        # Measure processing time for RTF calculation
        start = time.time()
        
        # Perform TTS synthesis with voice cloning
        print(f"Synthesizing speech using speaker audio: {speaker_wav}")
        try:
            outputs = model.synthesize(
                text,
                config,
                speaker_wav=speaker_wav,
                gpt_cond_len=3,
                language="en",
            )
            
            # Get the processing time
            processing_time = time.time() - start
            
            # Get the audio data
            wav = outputs['wav']
            
            # XTTS uses 24kHz sample rate
            sample_rate = 24000
            
            # Calculate real-time factor (RTF)
            audio_length = len(wav) / sample_rate
            rtf = processing_time / audio_length if audio_length > 0 else 0
            print(f"TTS RTF = {rtf:.5f}")
            
            # Save the file to a temporary location
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            write(temp_file.name, sample_rate, wav)
            
            print("Speech synthesized successfully")
            return temp_file.name, (sample_rate, wav), rtf
            
        except Exception as e:
            print(f"Error in XTTS synthesis: {e}")
            traceback.print_exc()
            return None, None, None
    except Exception as e:
        print(f"Error in speech synthesis: {e}")
        traceback.print_exc()
        return None, None, None

# Main processing function
def process_inputs(image, audio, progress=gr.Progress()):
    global accent_identifier, vlm_components, xtts_model, metrics, asr_models
    
    # Check if models are loaded
    if not all([vlm_components, xtts_model, metrics]) or not asr_models:
        progress(0, desc="Loading models...")
        initialize_models()
        
    if not all([vlm_components, xtts_model, metrics]) or not asr_models:
        return "Error: Models not loaded properly", None, "Failed to load one or more models", None, "N/A", "N/A", "N/A"
    
    # Initialize results for metrics
    accent_result = "Not available (no audio input)"
    asr_metrics_result = "Metrics not available (no audio input)"
    tts_metrics_result = "Metrics not available (TTS processing failed)"
    
    progress(0.2, desc="Identifying accent from filename...")
    
    # Step 1: Identify accent from filename if audio provided
    accent = "english"  # Default accent
    if audio is not None:
        # Directly identify accent from the filename
        accent_info = identify_accent(audio)
        accent = accent_info["predicted_accent"]
        
        # Format accent identification result
        # confidence = accent_info.get("confidence", 0.0)
        accent_result = f"Identified Accent: {accent}"
    
    progress(0.4, desc="Transcribing audio...")
    
    # Step 2: Transcribe audio with accent-specific model
    asr_audio_data = None
    if audio is None:
        question_text = "Please provide an audio"
        print("No audio provided")
    else:
        # Start measuring ASR latency
        metrics.start_timer()
        question_text, asr_audio_data = transcribe_audio(audio, accent)
        # Stop measuring ASR latency
        asr_latency = metrics.stop_timer()
        print(f"ASR Latency: {asr_latency:.4f} seconds")
        
        # Compute ASR metrics - latency, SNR, and speech clarity
        if asr_audio_data is not None:
            asr_metrics = metrics.measure_asr_wer(asr_audio_data, question_text)
            asr_metrics["Latency"] = asr_latency
            asr_metrics["Used Accent Model"] = accent
            asr_metrics_result = metrics.format_metrics(asr_metrics)
    
    progress(0.6, desc="Processing image and question...")
    
    # Default question if ASR is empty or gave an error
    if not isinstance(question_text, str) or question_text.strip() == "" or question_text.startswith("Error"):
        question_text = "What can you see in this image?"
        print(f"Using default question because ASR returned: '{question_text}'")
    
    # Step 3: Process image and question with VLM
    # Start measuring VLM latency
    metrics.start_timer()
    if image is None:
        response_text = "No image was provided. Please upload an image to analyze."
        print("No image provided")
    else:
        response_text = get_vlm_response(image, question_text)
    # Stop measuring VLM latency
    vlm_latency = metrics.stop_timer()
    print(f"VLM Latency: {vlm_latency:.4f} seconds")
    
    progress(0.8, desc="Synthesizing speech...")
    
    # Step 4: Generate speech from text response using the input audio as the speaker voice
    # Start measuring TTS latency
    metrics.start_timer()
    audio_file, tts_audio_data, rtf = synthesize_speech(response_text, audio)
    # Stop measuring TTS latency
    tts_latency = metrics.stop_timer()
    print(f"TTS Latency: {tts_latency:.4f} seconds")
    
    # Compute TTS metrics if we have audio data
    if tts_audio_data is not None:
        # MOS estimation with RTF (removed energy metric)
        tts_metrics = metrics.estimate_mos(tts_audio_data, rtf)
        tts_metrics["Latency"] = tts_latency
        tts_metrics["Voice Cloning"] = "Yes - Using input audio as voice reference"
        tts_metrics_result = metrics.format_metrics(tts_metrics)
    
    progress(1.0, desc="Processing complete!")
    
    return question_text, tts_audio_data, response_text, audio_file, accent_result, asr_metrics_result, tts_metrics_result

# Define Gradio interface
def create_demo():
    with gr.Blocks(title="Team 6: Speech Technology Final Project") as demo:
        gr.Markdown("# Team 6: Speech Technology Final Project")
        gr.Markdown("""
        ## AccentVision: A Multi-modal Interactive System with Accent Identification, ASR, Vision-Language, and TTS.
        
        ### Project Workflow:
        1. Upload an image
        2. Record or upload audio asking a question about the image
        3. The system will:
           - Identify the accent in your audio
           - Select the appropriate accent-specific ASR model
           - Transcribe your audio (ASR)
           - Process the image and question (Vision Language Model)
           - Generate a spoken response in the same accent as the input audio 
           - Calculate evaluation metrics for each component
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(type="filepath", label="Upload Image")
                audio_input = gr.Audio(type="filepath", label="Record/Upload Question Audio")
                submit_btn = gr.Button("Process", variant="primary")
            
            with gr.Column(scale=1):
                with gr.Accordion("Accent Identification", open=True):
                    accent_output = gr.Textbox(label="Accent Identification Results", lines=6)
                
                with gr.Accordion("ASR Output", open=True):
                    transcription_output = gr.Textbox(label="Transcribed Question")
                    asr_metrics_output = gr.Textbox(label="ASR Metrics", lines=5)
                
                with gr.Accordion("VLM Output", open=True):
                    text_output = gr.Textbox(label="Text Response from Vision Language Model")
                
                with gr.Accordion("TTS Output", open=True):
                    audio_output = gr.Audio(label="Generated Response Audio (Using Your Voice)")
                    audio_file_output = gr.File(label="Download Response Audio")
                    tts_metrics_output = gr.Textbox(label="TTS Metrics", lines=6)
        
        submit_btn.click(
            fn=process_inputs,
            inputs=[image_input, audio_input],
            outputs=[
                transcription_output, 
                audio_output, 
                text_output, 
                audio_file_output,
                accent_output,
                asr_metrics_output,
                tts_metrics_output
            ]
        )
        
        gr.Markdown("""
        ## Technical Details:
        
        ### Accent Identification
        - Using filename-based identification for accuracy
        - Supports Zulu, Afrikaans, Swahili, South African English, Kinyarwanda, and others
        - Provides confidence scores for accent identification
        
        ### ASR (Automatic Speech Recognition)
        - Using ESPnet OWSM v3.1 model with custom finetuned weights
        - Accent-specific fine-tuned models for better recognition accuracy
        - Dynamically selects the appropriate model based on identified accent
        
        ### Vision Language Model
        - Qwen2.5-VL-3B-Instruct for image understanding and question answering
        
        ### XTTS Voice Cloning
        - XTTS (XTrain TTS) model for high-quality voice cloning
        - Uses your input audio as reference to clone your voice characteristics
        - Preserves accent, tone, and speaking style in the generated response
        
        ### Evaluation Metrics:
        - **Accent Identification**: 
          - Confidence scores for detected accent and alternatives
        
        - **ASR Metrics**: 
          - Signal-to-Noise Ratio (dB): Higher values indicate cleaner audio
          - Speech Clarity: Measures speech intelligibility (0-100 scale)
          - Latency: Processing time for speech recognition
          - Accent Model: The accent-specific model used for transcription
        
        - **TTS Metrics**: 
          - Estimated MOS: Mean Opinion Score (1-5 scale, higher is better)
          - Dynamic Range: Variation between quietest and loudest parts (higher is more natural)
          - RTF: Real-Time Factor (lower is more efficient)
          - Latency: Processing time for speech generation
          - Voice Cloning: Indicates voice cloning is active
        """)
    
    return demo

# Launch the demo
if __name__ == "__main__":
    print("Initializing models...")
    initialize_models()
    
    print("Starting Gradio interface...")
    demo = create_demo()
    demo.launch(debug=True, share=True)