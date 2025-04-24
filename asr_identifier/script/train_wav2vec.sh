#!/bin/bash
#SBATCH --job-name=accent_recognizer
#SBATCH --output=accent_recognizer.out
#SBATCH --error=accent_recognizer.err
#SBATCH --partition=shire-general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16  # Adjusted to match the number of workers in your Python script
#SBATCH --gres=gpu:A100_80GB:1  # Requesting one A100 80GB GPU
#SBATCH --mem=16G  # Memory allocation as per your interactive command
#SBATCH --time=48:00:00  # Adjusted to match your interactive command

# Load necessary modules or activate environments
source /home/mkapadni/.bashrc  # Ensure this points to the correct file if different
conda activate gpu_env

# Navigate to the project directory
cd /home/mkapadni/work/speech_project/asr_identifier


python trainwav2vec2_base.py