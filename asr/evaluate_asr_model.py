# python inference_afrispeech.py --language english --finetuned_model /path/to/your/model.pth


import torch
import datasets
import numpy as np
import os
import soundfile as sf
import pandas as pd
import jiwer
import argparse
from espnet2.bin.s2t_inference import Speech2Text
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned OWSM model on AfriSpeech dataset")
    parser.add_argument("--language", type=str, default="igbo", 
                        choices=["afrikaans", "english", "hausa", "swahili", "igbo", 
                                "zulu", "south-african-english", "xhosa", 
                                "kinyarwanda", "hausa-fulani"],
                        help="Language from AfriSpeech dataset")
    parser.add_argument("--pretrained_model", type=str, default="espnet/owsm_v3.1_ebf_base",
                        help="Pretrained model name or path")
    parser.add_argument("--finetuned_model", type=str, default="/data/user_data/mkapadni/speech_project/exp/igbo/finetune/5epoch.pth",
                        help="Path to fine-tuned model checkpoint")
    parser.add_argument("--owsm_language", type=str, default="eng",
                        help="Language token for OWSM model")
    parser.add_argument("--output_dir", type=str, default="/data/user_data/mkapadni/speech_project/exp/igbo/output",
                        help="Directory to save outputs")
    parser.add_argument("--local_audio_dir", type=str, default="/data/user_data/mkapadni/speech_project/exp/igbo/output/saved_audio",
                        help="Directory to save audio files")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split to use for inference")
    
    return parser.parse_args()

def write_kaldi_files(dataset, output_dir, local_audio_dir):
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
            # Use the audio_id field as the utterance ID.
            utt_id = record["audio_id"]
            transcript = record["transcript"]
            speaker_id = record["speaker_id"]
            filename = os.path.basename(record["path"])
            audio_path = os.path.join(local_audio_dir, filename)

            # Write the transcription file: uttid <transcript>
            text_file.write(f"{utt_id} {transcript}\n")
            # Write the wav.scp file: uttid <audio_file_path>
            wav_scp_file.write(f"{utt_id} {audio_path}\n")
            # Write the utt2spk file: uttid speaker_id
            utt2spk_file.write(f"{utt_id} {speaker_id}\n")

    # Generate the spk2utt file by inverting utt2spk.
    utt2spk_map = {}
    for record in dataset:
        utt_id = record["audio_id"]
        speaker_id = record["speaker_id"]
        utt2spk_map.setdefault(speaker_id, []).append(utt_id)

    spk2utt_path = os.path.join(output_dir, "spk2utt")
    with open(spk2utt_path, "w", encoding="utf-8") as spk2utt_file:
        for speaker_id, utt_ids in utt2spk_map.items():
            spk2utt_file.write(f"{speaker_id} {' '.join(utt_ids)}\n")

    return text_path, wav_scp_path

def save_audio_files(dataset, local_audio_dir):
    """
    Save audio files from the dataset to a local directory.
    """
    os.makedirs(local_audio_dir, exist_ok=True)
    
    for record in dataset:
        # Access the audio dictionary; this will automatically decode and resample the audio.
        audio_info = record["audio"]
        audio_array = audio_info["array"]
        sampling_rate = audio_info["sampling_rate"]

        # Construct a local filename using the path field
        filename = os.path.basename(record["path"])
        local_audio_path = os.path.join(local_audio_dir, filename)
        
        # Save the audio file as a .wav file.
        sf.write(local_audio_path, audio_array, sampling_rate)

def load_model(pretrained_model_name, owsm_language, finetuned_model_path):
    """
    Load the pretrained model and fine-tuned weights.
    """
    # Load the pre-trained model
    pretrained_model = Speech2Text.from_pretrained(
        pretrained_model_name,
        lang_sym=f"<{owsm_language}>",
        beam_size=1,
        device='cuda'
    )
    
    # Make sure we use GPU
    pretrained_model.s2t_model.cuda()
    pretrained_model.device = 'cuda'
    
    # Load fine-tuned weights
    d = torch.load(finetuned_model_path)
    pretrained_model.s2t_model.load_state_dict(d)
    
    return pretrained_model

def run_inference(pretrained_model, wav_scp_path, text_path):
    """
    Run inference on audio files and return results.
    """
    # Load wav.scp
    wav_mapping = {}  # {speaker_id: wav_path}
    with open(wav_scp_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)  # Ensure only first space is used for split
            if len(parts) == 2:
                speaker_id, wav_path = parts
                wav_mapping[speaker_id] = wav_path

    # Load reference text
    reference_texts = {}  # {speaker_id: reference_text}
    with open(text_path, "r") as f:
        for line in f:
            parts = line.strip().split(" ", 1)
            if len(parts) == 2:
                speaker_id, ref_text = parts
                reference_texts[speaker_id] = ref_text

    results = []

    for speaker_id, wav_path in tqdm(wav_mapping.items(), total = len(wav_mapping.items())):
        if speaker_id not in reference_texts:
            continue  # Skip if reference text is missing

        # Load and process .wav file using soundfile
        try:
            waveform, sample_rate = sf.read(wav_path)
        except Exception as e:
            print(f"Error reading {wav_path}: {e}")
            continue

        # Resample if needed (ESPnet expects 16kHz)
        expected_sample_rate = 16000
        if sample_rate != expected_sample_rate:
            # Resampling using numpy for compatibility
            resample_factor = expected_sample_rate / sample_rate
            waveform = np.interp(np.arange(0, len(waveform), resample_factor), np.arange(len(waveform)), waveform)
            sample_rate = expected_sample_rate  # Update sample rate to match expected

        # Run ASR inference
        pred = pretrained_model(waveform)
        predicted_text = pred[0][0] if pred else ""

        # Store results
        results.append({
            "speaker_id": speaker_id,
            "reference_text": reference_texts[speaker_id],
            "predicted_text": predicted_text
        })
    
    return results

def main():
    args = parse_args()
    
    # Load the AfriSpeech dataset
    print(f"Loading {args.language} dataset from AfriSpeech-200")
    dataset = datasets.load_dataset(
        "intronhealth/afrispeech-200",
        args.language,
        split=args.split,
        trust_remote_code=True
    )
    
    # Cast to audio format with 16kHz sampling rate
    dataset = dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
    # Save audio files locally
    print(f"Saving audio files to {args.local_audio_dir}")
    save_audio_files(dataset, args.local_audio_dir)
    
    # Create output directory for split
    split_dir = os.path.join(args.output_dir, args.split)
    os.makedirs(split_dir, exist_ok=True)
    
    # Write Kaldi-style files
    print(f"Creating Kaldi-style files in {split_dir}")
    text_path, wav_scp_path = write_kaldi_files(dataset, split_dir, args.local_audio_dir)
    
    # Load model
    print(f"Loading pretrained model: {args.pretrained_model}")
    print(f"Loading fine-tuned weights from: {args.finetuned_model}")
    model = load_model(args.pretrained_model, args.owsm_language, args.finetuned_model)
    
    # Run inference
    print("Running inference...")
    results = run_inference(model, wav_scp_path, text_path)
    
    # Convert to DataFrame
    df_results = pd.DataFrame(results)
    
    # Compute overall WER
    wer = jiwer.wer(df_results["reference_text"].tolist(), df_results["predicted_text"].tolist())
    
    # Save results to CSV
    results_path = os.path.join(args.output_dir, f"{args.language}_{args.split}_asr_results.csv")
    df_results.to_csv(results_path, index=False)
    
    print(f"Results saved to {results_path}")
    print(f"Overall WER for {args.language} ({args.split} split): {wer:.4f}")

if __name__ == "__main__":
    main()