import torch
import datasets
import espnetez as ez
import numpy as np
import librosa
from espnet2.bin.s2t_inference import Speech2Text
import wandb
wandb.init(settings=wandb.Settings(init_timeout=300))

import os
os.environ["HF_HOME"] = "/home/mkapadni/.cache"

model_dir = os.path.join(os.environ["HF_HOME"], "models--espnet--owsm_v3.1_ebf")
if os.path.exists(model_dir):
    import shutil
    shutil.rmtree(model_dir)


# Step 1: Load the AfriSpeech dataset with proper splits
# Make sure to use the correct dataset ID: "intronhealth/afrispeech-200" instead of "tobiolatunji/afrispeech-200" 

LANGUAGE = 'english'

train_dataset = datasets.load_dataset("intronhealth/afrispeech-200", LANGUAGE, split="validation")
train_dataset = train_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

# valid_dataset = datasets.load_dataset("intronhealth/afrispeech-200", "english", split="validation")
# valid_dataset = valid_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

test_dataset = datasets.load_dataset("intronhealth/afrispeech-200", LANGUAGE, split="test")
test_dataset = test_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))

# Step 2: Load the pretrained model
FINETUNE_MODEL = "espnet/owsm_v3.1_ebf"
owsm_language = "eng"  # this remains same always

pretrained_model = Speech2Text.from_pretrained(
    FINETUNE_MODEL,
    lang_sym=f"<{owsm_language}>",
    # use_flash_attn=False,
    # task_sym='<asr>',
    beam_size=1,
    device='cuda',
    # cache_dir= "/data/user_data/mkapadni/hf_cache/models"
)

torch.save(pretrained_model.s2t_model.state_dict(), 'original.pth')
pretrain_config = vars(pretrained_model.s2t_train_args)
tokenizer = pretrained_model.tokenizer
converter = pretrained_model.converter

def get_audio_array(d):
    if isinstance(d, dict) and "audio" in d:
        if isinstance(d["audio"], dict) and "array" in d["audio"]:
            return d["audio"]["array"]
        elif hasattr(d["audio"], "array"):
            return d["audio"].array
        return d["audio"]
    # If it's a tuple
    elif isinstance(d, tuple):
        # Try the first element, which typically contains audio
        if len(d) > 0:
            if hasattr(d[0], "array"):
                return d[0].array
            return d[0]
    # Fall back to empty array
    return np.array([], dtype=np.float32)

def get_transcript(d):
    if isinstance(d, dict) and "transcript" in d:
        return d["transcript"]
    # If it's a tuple
    elif isinstance(d, tuple) and len(d) > 1:
        # Transcript is often the second element
        return d[1]
    # Fall back to empty string
    return ""

# Define tokenization function that converts to Long type
def tokenize(text):
    # Ensure we convert to int64/long which is required for embedding layers
    return np.array(converter.tokens2ids(tokenizer.text2tokens(text)), dtype=np.int64)

# Define data_info with robust functions
data_info = {
    "speech": lambda d: get_audio_array(d).astype(np.float32),
    "text": lambda d: tokenize(f"<{owsm_language}> {get_transcript(d)}"),
    "text_prev": lambda d: tokenize(""),
    "text_ctc": lambda d: tokenize(get_transcript(d)),
}
# Create ESPnetEZDataset instances
train_ez_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
# valid_ez_dataset = ez.dataset.ESPnetEZDataset(valid_dataset, data_info=data_info)
test_ez_dataset = ez.dataset.ESPnetEZDataset(test_dataset, data_info=data_info)


# Step 6: Define model loading function
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_model_fn(args):
    model = pretrained_model.s2t_model
    model.train()
    print(f'Trainable parameters: {count_parameters(model)}')
    return model

# Step 7: Set up experiment directories
# need to change this languagewise itself
EXP_DIR = f"/data/user_data/mkapadni/speech_project/exp/{LANGUAGE}/normal_model_finetune"
STATS_DIR = f"/data/user_data/mkapadni/speech_project/exp/{LANGUAGE}/normal_model_stats_finetune"

# Step 8: Update finetune config
finetune_config = ez.config.update_finetune_config(
    's2t',
    pretrain_config,
    f"./config/finetune.yaml"
)

# Customize config for AfriSpeech dataset
finetune_config['max_epoch'] = 5  # Start with a smaller number for testing
finetune_config['num_iters_per_epoch'] = 500
finetune_config['batch_size'] = 4  # Reduce batch size if you encounter memory issues
finetune_config['accum_grad'] = 4  # Gradient accumulation steps

# Update your finetune config to ensure consistent data types
finetune_config['dtype'] = 'float32'  # Use float32 for model parameters
finetune_config['token_type'] = 'int64'  # Use int64 for token indices
finetune_config['use_wandb'] = "true"

# Step 9: Create trainer with correct datasets
trainer = ez.Trainer(
    task='s2t',
    train_config=finetune_config,
    train_dataset=train_ez_dataset,
    valid_dataset=test_ez_dataset,  # Use valid_ez_dataset instead of test_ez_dataset
    build_model_fn=build_model_fn,
    data_info=data_info,
    output_dir=EXP_DIR,
    stats_dir=STATS_DIR,
    ngpu=1
)

# Step 10: Collect stats and train
trainer.collect_stats()  # This should now work

trainer.train()