#!/bin/bash
#SBATCH --job-name=SA_english
#SBATCH --output=SA_english.out
#SBATCH --error=SA_english.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=12:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/speech_project/tts_inference


python evaluate_tts.py \
  --language south-african-english \
  --split test \
  --finetuned_model "/data/user_data/mkapadni/hf_cache/hub/models--intronhealth--afro-tts/snapshots/f99ba77006ffcf2fb66ed12d7093832b552a3d26" \
  --reference_wav "/data/user_data/mkapadni/hf_cache/hub/models--intronhealth--afro-tts/snapshots/f99ba77006ffcf2fb66ed12d7093832b552a3d26/audios/reference_accent.wav" \
  --gpt_cond_len 3 \
  --tts_language "en" \
  --sample_rate 24000 \
  --output_dir "/data/user_data/mkapadni/speech_project/tts_outputs_SA_english" 
