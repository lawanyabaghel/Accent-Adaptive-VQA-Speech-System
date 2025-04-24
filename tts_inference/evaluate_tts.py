import os
import time
import argparse
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from datasets import load_dataset
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
# Existing evaluation imports
from transformers import pipeline
from jiwer import wer, cer
# New imports for audio-based metrics
from python_speech_features import mfcc
import librosa


def parse_args():
    p = argparse.ArgumentParser(
        description="Run TTS inference with a fine‐tuned Xtts model on AfriSpeech and evaluate results"
    )
    p.add_argument("--language", type=str, default="english",
                   choices=["afrikaans","english","hausa","swahili","igbo",
                            "zulu","south-african-english","xhosa","tswana",
                            "kinyarwanda","isizulu"],
                   help="Language split from AfriSpeech‑200")
    p.add_argument("--split", type=str, default="test",
                   choices=["train","validation","test"],
                   help="Which split to run TTS on")
    p.add_argument("--finetuned_model", type=str, default="/data/user_data/mkapadni/hf_cache/hub/models--intronhealth--afro-tts/snapshots/f99ba77006ffcf2fb66ed12d7093832b552a3d26",
                   help="Directory containing config.json + checkpoint files")
    p.add_argument("--reference_wav", type=str, default="/data/user_data/mkapadni/hf_cache/hub/models--intronhealth--afro-tts/snapshots/f99ba77006ffcf2fb66ed12d7093832b552a3d26/audios/reference_accent.wav",
                   help="Path to a reference speaker wav for style conditioning")
    p.add_argument("--gpt_cond_len", type=int, default=3,
                   help="Number of GPT conditioning frames")
    p.add_argument("--tts_language", type=str, default="en",
                   help="Language code for XttsModel (e.g. 'en')")
    p.add_argument("--sample_rate", type=int, default=24000,
                   help="Output sampling rate")
    p.add_argument("--output_dir", type=str, default="/data/user_data/mkapadni/speech_project/tts_outputs",
                   help="Where to save generated wavs and metrics CSV")
    p.add_argument("--local_audio_dir", type=str, default="/data/user_data/mkapadni/speech_project/saved_audio",
                   help="Directory containing original dataset audio files")
    p.add_argument("--mos", type=float, default=None,
                   help="(Optional) Mean Opinion Score from human eval")
    # Evaluation arguments
    p.add_argument("--evaluate_only", action="store_true",
                   help="Skip TTS generation and only evaluate existing outputs")
    p.add_argument("--asr_model", type=str, default="openai/whisper-large-v2",
                   help="ASR model for evaluation metrics")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Maximum number of samples to evaluate")
    p.add_argument("--skip_asr", action="store_true",
                   help="Skip ASR-based evaluation")
    return p.parse_args()

def load_tts_model(model_dir):
    # load Xtts config + model
    cfg = XttsConfig()
    cfg_path = os.path.join(model_dir, "config.json")
    cfg.load_json(cfg_path)
    model = Xtts.init_from_config(cfg)
    model.load_checkpoint(cfg, checkpoint_dir=model_dir, eval=True)
    model.cuda()
    return cfg, model

def run_inference(args, cfg, model):
    # load transcripts
    dataset = load_dataset(
        "intronhealth/afrispeech-200",
        args.language,
        split=args.split,
        trust_remote_code=True
    )
    # we'll only need the text
    records = [(r["audio_id"], r["transcript"]) for r in dataset]

    os.makedirs(args.output_dir, exist_ok=True)
    rows = []

    for utt_id, text in records:
        start = time.perf_counter()
        outputs = model.synthesize(
            text,
            cfg,
            speaker_wav=args.reference_wav,
            gpt_cond_len=args.gpt_cond_len,
            language=args.tts_language,
            enable_text_splitting=True,
        )
        end = time.perf_counter()
        wav = outputs["wav"]
        inf_time = end - start
        duration = len(wav) / args.sample_rate
        rtf = inf_time / duration if duration > 0 else float("nan")
        dyn_range = (wav.max() - wav.min()) / 2.0

        # Create directory if needed (for nested paths)
        out_path = os.path.join(args.output_dir, f"{utt_id}_tts.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # save
        sf.write(out_path, wav, args.sample_rate)

        rows.append({
            "utt_id": utt_id,
            "text": text,
            "inference_time_s": inf_time,
            "duration_s": duration,
            "rtf": rtf,
            "dynamic_range": dyn_range,
            "wav_path": out_path,
        })

        print(f"[{utt_id}] inf_time {inf_time:.3f}s | dur {duration:.3f}s | "
              f"RTF {rtf:.3f} | dyn_rng {dyn_range:.3f}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.output_dir, f"{args.language}_{args.split}_tts_metrics.csv")
    df.to_csv(csv_path, index=False)
    return df, csv_path, dataset

def truncate_tokens(model, text, language="en", max_tokens=400):
    """Truncate input text to ensure it doesn't exceed the token limit."""
    text_tokens = model.tokenizer.encode(text, lang=language)
    
    # Check if truncation is needed
    if len(text_tokens) >= max_tokens:
        # Truncate tokens
        truncated_tokens = text_tokens[:max_tokens-1]
        # Convert back to text
        truncated_text = model.tokenizer.decode(truncated_tokens)
        print(f"Text truncated from {len(text_tokens)} to {len(truncated_tokens)} tokens")
        return truncated_text
    
    return text


# Function to compute Mel Cepstral Distortion (MCD)
def compute_mcd(wav_ref, wav_pred, sr=24000):
    """Compute Mel Cepstral Distortion between reference and predicted audio"""
    # Extract MFCC features
    mfcc_ref = mfcc(wav_ref, samplerate=sr, numcep=13)
    mfcc_pred = mfcc(wav_pred, samplerate=sr, numcep=13)
    
    # Align lengths
    min_len = min(len(mfcc_ref), len(mfcc_pred))
    mfcc_ref = mfcc_ref[:min_len]
    mfcc_pred = mfcc_pred[:min_len]
    
    # Compute Euclidean distance frame-wise
    dist = np.linalg.norm(mfcc_ref - mfcc_pred, axis=1)
    
    # MCD formula (in dB)
    mcd = (10.0 / np.log(10)) * np.mean(dist)
    return mcd

# Placeholder functions for PESQ and STOI metrics
def compute_pesq(wav_ref, wav_pred, sr=24000):
    """
    Compute Perceptual Evaluation of Speech Quality (PESQ) between reference and predicted audio.
    
    PESQ is an objective metric that evaluates the quality of speech signals, returning
    a score between -0.5 and 4.5, where higher scores indicate better quality.
    
    Args:
        wav_ref: Reference/clean audio waveform
        wav_pred: Predicted/processed audio waveform
        sr: Sample rate of the input audio (will be resampled to 16kHz if different)
        
    Returns:
        float: PESQ score or np.nan if calculation fails
    """
    # Try to import pesq from different possible packages
    try:
        from pesq import pesq
        pesq_fn = pesq
    except ImportError:
        try:
            from pypesq import pypesq
            pesq_fn = pypesq
        except ImportError:
            print("PESQ metric not available. Please install with: pip install pesq")
            return np.nan
    
    # PESQ only supports 8000Hz or 16000Hz sample rates
    target_sr = 16000  # Use wideband mode with 16kHz
    
    try:
        # Ensure arrays are float type for resampling
        wav_ref = wav_ref.astype(np.float32)
        wav_pred = wav_pred.astype(np.float32)
        
        # Resample if needed
        if sr != target_sr:
            wav_ref = librosa.resample(wav_ref, orig_sr=sr, target_sr=target_sr)
            wav_pred = librosa.resample(wav_pred, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio (recommended for PESQ)
        wav_ref = wav_ref / (np.max(np.abs(wav_ref)) + 1e-8)
        wav_pred = wav_pred / (np.max(np.abs(wav_pred)) + 1e-8)
        
        # Match lengths (PESQ requires same length signals)
        min_len = min(len(wav_ref), len(wav_pred))
        wav_ref = wav_ref[:min_len]
        wav_pred = wav_pred[:min_len]
        
        # Calculate PESQ score
        score = pesq_fn(target_sr, wav_ref, wav_pred, 'wb')  # wideband mode
        return score
    
    except Exception as e:
        print(f"PESQ computation failed: {e}")
        return np.nan

def compute_stoi(wav_ref, wav_pred, sr=24000):
    """
    Compute Short-Time Objective Intelligibility (STOI) between reference and predicted audio.
    
    STOI is an intrusive metric that evaluates speech intelligibility, returning
    a score between 0 and 1, where higher scores indicate better intelligibility.
    
    Args:
        wav_ref: Reference/clean audio waveform
        wav_pred: Predicted/processed audio waveform
        sr: Sample rate of the input audio (will be resampled to 16kHz if different)
        
    Returns:
        float: STOI score or np.nan if calculation fails
    """
    # Try to import stoi
    try:
        from pystoi import stoi
        stoi_fn = stoi
    except ImportError:
        print("STOI metric not available. Please install with: pip install pystoi")
        return np.nan
    
    # STOI works best with 16kHz audio
    target_sr = 16000
    
    try:
        # Ensure arrays are float type for resampling
        wav_ref = wav_ref.astype(np.float32)
        wav_pred = wav_pred.astype(np.float32)
        
        # Resample if needed
        if sr != target_sr:
            wav_ref = librosa.resample(wav_ref, orig_sr=sr, target_sr=target_sr)
            wav_pred = librosa.resample(wav_pred, orig_sr=sr, target_sr=target_sr)
        
        # Normalize audio (generally a good practice)
        wav_ref = wav_ref / (np.max(np.abs(wav_ref)) + 1e-8)
        wav_pred = wav_pred / (np.max(np.abs(wav_pred)) + 1e-8)
        
        # Match lengths (STOI requires same length signals)
        min_len = min(len(wav_ref), len(wav_pred))
        wav_ref = wav_ref[:min_len]
        wav_pred = wav_pred[:min_len]
        
        # Calculate STOI score (non-extended version)
        score = stoi_fn(wav_ref, wav_pred, target_sr, extended=False)
        return score
    
    except Exception as e:
        print(f"STOI computation failed: {e}")
        return np.nan

# New function to evaluate using ground truth audio
def evaluate_with_ground_truth(args, df, dataset):
    """Compare synthesized audio with ground truth audio using objective metrics"""
    print("\nRunning TTS evaluation using ground truth audio comparison...")
    
    # Add columns for ground truth audio metrics
    df["gt_wav_path"] = ""
    df["mcd"] = np.nan
    df["pesq"] = np.nan
    df["stoi"] = np.nan
    
    # Create mapping from utterance ID to ground truth audio path
    utt_to_gt_audio = {}
    for record in dataset:
        utt_id = record["audio_id"]
        filename = os.path.basename(record["path"])
        audio_path = os.path.join(args.local_audio_dir, filename)
        utt_to_gt_audio[utt_id] = audio_path
    
    # Limit samples if specified
    if args.max_samples:
        evaluate_df = df.head(args.max_samples)
    else:
        evaluate_df = df
    
    for idx, row in evaluate_df.iterrows():
        try:
            utt_id = row["utt_id"]
            pred_path = row["wav_path"]
            
            # Get ground truth audio path
            if utt_id in utt_to_gt_audio:
                gt_path = utt_to_gt_audio[utt_id]

                df.at[idx, "gt_wav_path"] = gt_path
            else:
                print(f"Warning: No ground truth audio found for {utt_id}")
                continue
                
            if not os.path.exists(gt_path) or not os.path.exists(pred_path):
                print(f"Warning: Missing audio files for {utt_id}")
                continue
                
            print(f"Comparing: {os.path.basename(pred_path)} with ground truth")
            
            # Load audio files
            wav_gt, sr_gt = sf.read(gt_path)
            wav_pred, sr_pred = sf.read(pred_path)
            
            # Resample if needed
            if sr_gt != args.sample_rate:
                print(f"  Warning: Ground truth sample rate mismatch ({sr_gt} vs {args.sample_rate})")
                # For simplicity, we'll just notify about the mismatch
            
            # Compute MCD
            mcd_val = compute_mcd(wav_gt, wav_pred, sr=args.sample_rate)
            
            # Update dataframe
            df.at[idx, "mcd"] = mcd_val
            df.at[idx, "pesq"] = compute_pesq(wav_gt, wav_pred, sr=args.sample_rate)
            df.at[idx, "stoi"] = compute_stoi(wav_gt, wav_pred, sr=args.sample_rate)
            
            print(f"  MCD: {mcd_val:.4f} (lower is better)")
            
        except Exception as e:
            print(f"Error evaluating {row['utt_id']} with ground truth: {e}")
    
    # Save updated metrics
    eval_csv_path = os.path.join(args.output_dir, f"{args.language}_{args.split}_gt_evaluation.csv")
    df.to_csv(eval_csv_path, index=False)
    print(f"Ground truth evaluation metrics saved to {eval_csv_path}")
    
    return df

# Original ASR evaluation function
def evaluate_with_asr(args, df):
    """Evaluate the generated TTS audio using ASR for WER/CER metrics."""
    print("\nRunning TTS evaluation using ASR...")
    asr = load_asr_model(args.asr_model)
    
    # Add evaluation metric columns
    df["asr_transcript"] = ""
    df["wer"] = np.nan
    df["cer"] = np.nan
    
    # Limit samples if specified
    if args.max_samples:
        evaluate_df = df.head(args.max_samples)
    else:
        evaluate_df = df
    
    for idx, row in evaluate_df.iterrows():
        try:
            if not os.path.exists(row["wav_path"]):
                print(f"Warning: File not found: {row['wav_path']}")
                continue
                
            print(f"Evaluating: {os.path.basename(row['wav_path'])}")
            # Run ASR on the synthesized audio
            result = asr(row["wav_path"])
            asr_text = result["text"]
            
            # Calculate WER and CER
            reference = row["text"].lower()
            hypothesis = asr_text.lower()
            
            current_wer = wer(reference, hypothesis)
            current_cer = cer(reference, hypothesis)
            
            # Update the dataframe
            df.at[idx, "asr_transcript"] = asr_text
            df.at[idx, "wer"] = current_wer
            df.at[idx, "cer"] = current_cer
            
            print(f"  Original text: {reference}")
            print(f"  ASR transcription: {asr_text}")
            print(f"  WER: {current_wer:.4f}, CER: {current_cer:.4f}")
            
        except Exception as e:
            print(f"Error evaluating {row['wav_path']}: {e}")
    
    # Save updated metrics
    eval_csv_path = os.path.join(args.output_dir, f"{args.language}_{args.split}_asr_evaluation.csv")
    df.to_csv(eval_csv_path, index=False)
    print(f"ASR evaluation metrics saved to {eval_csv_path}")
    
    return df

# Helper function to load ASR model
def load_asr_model(model_name):
    """Load ASR model for evaluation."""
    print(f"Loading ASR model {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    asr = pipeline("automatic-speech-recognition", model=model_name, device=device)
    return asr

# Function to load existing metrics
def load_existing_metrics(args):
    """Load previously generated metrics if only evaluating."""
    csv_path = os.path.join(args.output_dir, f"{args.language}_{args.split}_tts_metrics.csv")
    if os.path.exists(csv_path):
        print(f"Loading existing metrics from {csv_path}")
        return pd.read_csv(csv_path), csv_path
    else:
        raise FileNotFoundError(f"Cannot find metrics file at {csv_path}. Run without --evaluate_only first.")

# Function to load dataset
def load_evaluation_dataset(args):
    """Load the dataset for evaluation."""
    print(f"Loading {args.language}/{args.split} dataset for evaluation...")
    dataset = load_dataset(
        "intronhealth/afrispeech-200",
        args.language,
        split=args.split,
        trust_remote_code=True
    )
    print(f"Loaded {len(dataset)} records from the dataset")
    return dataset

def summarize(df, args):
    """Print summary of all evaluation metrics."""
    # Reference-free metrics summary
    avg_lat = df["inference_time_s"].mean() if "inference_time_s" in df.columns else np.nan
    avg_rtf = df["rtf"].mean() if "rtf" in df.columns else np.nan
    avg_dyn = df["dynamic_range"].mean() if "dynamic_range" in df.columns else np.nan
    
    print("\n=== TTS REFERENCE-FREE METRICS SUMMARY ===")
    if args.mos is not None:
        print(f"Mean Opinion Score (MOS): {args.mos:.2f}")
    else:
        print("Mean Opinion Score (MOS): N/A (provide --mos or use a MOS predictor)")
    print(f"Dynamic Range: {avg_dyn:.4f}")
    print(f"Real‑Time Factor (RTF): {avg_rtf:.4f}")
    print(f"Latency (avg inference time): {avg_lat:.4f} s")
    
    # ASR-based metrics summary
    if "wer" in df.columns and not df["wer"].isna().all():
        avg_wer = df["wer"].mean()
        avg_cer = df["cer"].mean()
        print("\n=== ASR EVALUATION METRICS ===")
        print(f"Word Error Rate (WER): {avg_wer:.4f}")
        print(f"Character Error Rate (CER): {avg_cer:.4f}")
        print(f"Number of samples evaluated: {df['wer'].count()}")
    
    # Ground truth comparison metrics
    if "mcd" in df.columns and not df["mcd"].isna().all():
        avg_mcd = df["mcd"].mean()
        avg_pesq = df["pesq"].mean() if "pesq" in df.columns else np.nan
        avg_stoi = df["stoi"].mean() if "stoi" in df.columns else np.nan
        
        print("\n=== GROUND TRUTH AUDIO EVALUATION METRICS ===")
        print(f"Mel Cepstral Distortion (MCD): {avg_mcd:.4f} dB (lower is better)")
        
        if not np.isnan(avg_pesq):
            print(f"Perceptual Evaluation of Speech Quality (PESQ): {avg_pesq:.4f} (higher is better)")
        
        if not np.isnan(avg_stoi):
            print(f"Short-Time Objective Intelligibility (STOI): {avg_stoi:.4f} (higher is better)")
            
        print(f"Number of samples compared: {df['mcd'].count()}")


def main():
    args = parse_args()
    
    # Check if we need to run TTS inference
    if args.evaluate_only:
        print("Loading existing metrics (skipping TTS generation)...")
        df, csv_path = load_existing_metrics(args)
        # We still need to load the dataset for ground truth comparison
        dataset = load_evaluation_dataset(args)
    else:
        print("Loading TTS model...")
        cfg, model = load_tts_model(args.finetuned_model)
        print(f"Running TTS on {args.language}/{args.split}...")
        df, csv_path, dataset = run_inference(args, cfg, model)
        print(f"\nMetrics per utterance saved to {csv_path}")
    
    # Evaluate with ground truth audio
    print("Evaluating with ground truth audio...")
    df = evaluate_with_ground_truth(args, df, dataset)
    
    # Evaluate with ASR if not skipped
    if not args.skip_asr:
        df = evaluate_with_asr(args, df)
    
    # Print summary statistics
    summarize(df, args)

if __name__ == "__main__":
    main()
