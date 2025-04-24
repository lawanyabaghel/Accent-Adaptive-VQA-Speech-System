from scipy.io.wavfile import write
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
import os
from datasets import load_dataset
import soundfile as sf
from huggingface_hub import hf_hub_download

# Create output directories
local_audio_dir = "/data/user_data/mkapadni/speech_project/saved_audio"
output_dir = "/data/user_data/mkapadni/speech_project/data"
tts_output_dir = "/data/user_data/mkapadni/speech_project/tts_outputs"
os.makedirs(local_audio_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(tts_output_dir, exist_ok=True)

# Set the dataset split
split = "test"

# Check for CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Download model files from HuggingFace
print("Downloading model files from HuggingFace...")
try:
    config_path = hf_hub_download(repo_id="intronhealth/afro-tts", filename="config.json")
    print(f"Downloaded config from HuggingFace: {config_path}")
    
    # Download reference audio
    reference_audio_path = hf_hub_download(repo_id="intronhealth/afro-tts", filename="audios/reference_accent.wav")
    print(f"Downloaded reference audio: {reference_audio_path}")
except Exception as e:
    print(f"Error downloading from HuggingFace: {e}")
    raise

# Load model using XTTS approach
print("Loading model...")
config = XttsConfig()
config.load_json(config_path)
model = Xtts.init_from_config(config)

# The checkpoint_dir should be the directory containing the model files on HuggingFace
checkpoint_dir = os.path.dirname(config_path)
model.load_checkpoint(config, checkpoint_dir="/data/user_data/mkapadni/hf_cache/hub/models--intronhealth--afro-tts/snapshots/f99ba77006ffcf2fb66ed12d7093832b552a3d26", eval=True)
model.cuda()  # Move model to GPU
print("Model loaded successfully")

# Function to write Kaldi-style files
def write_kaldi_files(dataset, output_dir):
    """
    Create Kaldi-style files (text, wav.scp, utt2spk, spk2utt)
    from the AfriSpeech dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    text_path = os.path.join(output_dir, "text")
    wav_scp_path = os.path.join(output_dir, "wav.scp")
    utt2spk_path = os.path.join(output_dir, "utt2spk")
    
    with open(text_path, "w", encoding="utf-8") as text_file, \
         open(wav_scp_path, "w", encoding="utf-8") as wav_scp_file, \
         open(utt2spk_path, "w", encoding="utf-8") as utt2spk_file:
        for record in dataset:
            utt_id = record["audio_id"]
            transcript = record["transcript"]
            speaker_id = record["speaker_id"]
            
            filename = os.path.basename(record["path"])
            audio_path = os.path.join(local_audio_dir, filename)
            
            text_file.write(f"{utt_id} {transcript}\n")
            wav_scp_file.write(f"{utt_id} {audio_path}\n")
            utt2spk_file.write(f"{utt_id} {speaker_id}\n")
    
    # Generate the spk2utt file by inverting utt2spk
    utt2spk_map = {}
    for record in dataset:
        utt_id = record["audio_id"]
        speaker_id = record["speaker_id"]
        utt2spk_map.setdefault(speaker_id, []).append(utt_id)
    
    spk2utt_path = os.path.join(output_dir, "spk2utt")
    with open(spk2utt_path, "w", encoding="utf-8") as spk2utt_file:
        for speaker_id, utt_ids in utt2spk_map.items():
            spk2utt_file.write(f"{speaker_id} {' '.join(utt_ids)}\n")

# Function to read the speaker-text mapping
def read_speaker_text_file(file_path):
    speaker_texts = {}
    try:
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                speaker_id, text = line.split(' ', 1)  # Split at the first space
                speaker_texts[speaker_id] = text
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return speaker_texts

# Load the AfriSpeech dataset
print("Loading AfriSpeech dataset...")
dataset = load_dataset(
    "tobiolatunji/afrispeech-200",
    "english",
    split=split,
    trust_remote_code=True
)
print(f"Loaded {len(dataset)} records from the dataset")

# Save audio files from the dataset
print("Saving audio files from the dataset...")
for record in dataset:
    audio_info = record["audio"]
    audio_array = audio_info["array"]
    sampling_rate = audio_info["sampling_rate"]
    
    filename = os.path.basename(record["path"])
    local_audio_path = os.path.join(local_audio_dir, filename)
    print(f"Saving audio to: {local_audio_path}")
    
    sf.write(local_audio_path, audio_array, sampling_rate)

# Create a subdirectory for the split
split_dir = os.path.join(output_dir, split)
os.makedirs(split_dir, exist_ok=True)

# Write the Kaldi-style files
print("Creating Kaldi-style files...")
write_kaldi_files(dataset, split_dir)

# Read the reference texts
input_text_file = os.path.join(split_dir, "text")
reference_texts = read_speaker_text_file(input_text_file)
print(f"Loaded {len(reference_texts)} texts for speech synthesis")

# Synthesize speech for each text in the reference file
print("Starting speech synthesis...")
results = []
sample_rate = 24000  # Afro-TTS uses 24kHz sampling rate

for speaker_id, text_input in reference_texts.items():
    try:
        print(f"Synthesizing speech for speaker {speaker_id}...")
        
        # Generate speech using the XTTS model directly
        outputs = model.synthesize(
            text=text_input,
            config=config,
            speaker_wav=reference_audio_path,
            gpt_cond_len=3,
            language="en"
        )
        
        # Validate waveform
        if outputs['wav'] is None or len(outputs['wav']) == 0:
            print(f"Error: Invalid waveform for {speaker_id}")
            continue
        
        # Create output filename
        output_path = os.path.join(tts_output_dir, f"{speaker_id}_synth.wav")
        
        # Handle long filenames
        if len(output_path) > 255:
            print(f"Warning: File path too long for {speaker_id}, shortening.")
            output_path = os.path.join(tts_output_dir, f"{speaker_id[:20]}_synth.wav")
        
        # Ensure the file is not being overwritten
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(tts_output_dir, f"{speaker_id}_{counter}_synth.wav")
            counter += 1
        
        # Save the generated .wav file

        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Saving synthesized speech to: {output_path}")
        write(output_path, sample_rate, outputs['wav'])
        
        # Store results
        results.append({
            "speaker_id": speaker_id,
            "reference_text": text_input,
            "output_wav_path": output_path
        })
        
    except Exception as e:
        print(f"TTS synthesis failed for {speaker_id}: {e}")
        continue

print(f"Completed speech synthesis for {len(results)} speakers.")
