import torch
import datasets
import espnetez as ez
import numpy as np
import argparse
import os
from espnet2.bin.s2t_inference import Speech2Text

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune OWSM model on AfriSpeech dataset")
    parser.add_argument("--language", type=str, required=True, 
                        choices=["afrikaans", "english", "hausa", "swahili", "igbo", 
                                "zulu", "south-african-english", "xhosa", 
                                "kinyarwanda", "hausa-fulani"],
                        help="Language from AfriSpeech dataset")
    parser.add_argument("--pretrained_model", type=str, default="espnet/owsm_v3.1_ebf_base",
                        help="Pretrained model name or path")
    parser.add_argument("--owsm_language", type=str, default="eng",
                        help="Language token for OWSM model")
    parser.add_argument("--epochs", type=int, default=2,
                        help="Number of training epochs")
    parser.add_argument("--iterations", type=int, default=500,
                        help="Number of iterations per epoch")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--grad_accum", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save outputs")
    parser.add_argument("--stats_dir", type=str, default=None,
                        help="Directory to save stats")
    parser.add_argument("--config_path", type=str, default="./config/finetune.yaml",
                        help="Path to finetune config file")
    parser.add_argument("--ngpu", type=int, default=1,
                        help="Number of GPUs to use")
    parser.add_argument("--use_wandb", action="store_true",
                        help="Whether to use wandb for logging")
    
    return parser.parse_args()

def get_audio_array(d):
    if isinstance(d, dict) and "audio" in d:
        if isinstance(d["audio"], dict) and "array" in d["audio"]:
            return d["audio"]["array"]
        elif hasattr(d["audio"], "array"):
            return d["audio"].array
        return d["audio"]
    elif isinstance(d, tuple) and len(d) > 0:
        if hasattr(d[0], "array"):
            return d[0].array
        return d[0]
    return np.array([], dtype=np.float32)

def get_transcript(d):
    if isinstance(d, dict) and "transcript" in d:
        return d["transcript"]
    elif isinstance(d, tuple) and len(d) > 1:
        return d[1]
    return ""

def load_datasets(language):
    """
    Load dataset splits based on the language.
    English has dev/validation and test splits.
    Other languages have train and test splits.
    """
    print(f"Loading {language} dataset from AfriSpeech-200")
    
    # All languages have a test split
    test_dataset = datasets.load_dataset("intronhealth/afrispeech-200", language, split="test")
    test_dataset = test_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    
    # For English, use validation split for training
    # For other languages, use train split
    try:
        if language == "english":
            train_dataset = datasets.load_dataset("intronhealth/afrispeech-200", language, split="validation")
        else:
            train_dataset = datasets.load_dataset("intronhealth/afrispeech-200", language, split="train")
        
        train_dataset = train_dataset.cast_column("audio", datasets.Audio(sampling_rate=16000))
    except Exception as e:
        print(f"Error loading train/validation dataset: {e}")
        print("Falling back to test dataset for training")
        train_dataset = test_dataset
    
    return train_dataset, test_dataset

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    args = parse_args()
    
    # Load datasets based on language
    train_dataset, test_dataset = load_datasets(args.language)
    
    # Load pretrained model
    print(f"Loading pretrained model: {args.pretrained_model}")
    pretrained_model = Speech2Text.from_pretrained(
        args.pretrained_model,
        lang_sym=f"<{args.owsm_language}>",
        beam_size=1,
        device='cuda'
    )
    
    torch.save(pretrained_model.s2t_model.state_dict(), 'original.pth')
    pretrain_config = vars(pretrained_model.s2t_train_args)
    tokenizer = pretrained_model.tokenizer
    converter = pretrained_model.converter
    
    # Define tokenization function
    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)), dtype=np.int64)
    
    # Define data_info
    data_info = {
        "speech": lambda d: get_audio_array(d).astype(np.float32),
        "text": lambda d: tokenize(f"<{args.owsm_language}> {get_transcript(d)}"),
        "text_prev": lambda d: tokenize(""),
        "text_ctc": lambda d: tokenize(get_transcript(d)),
    }
    
    # Create ESPnetEZDataset instances
    train_ez_dataset = ez.dataset.ESPnetEZDataset(train_dataset, data_info=data_info)
    test_ez_dataset = ez.dataset.ESPnetEZDataset(test_dataset, data_info=data_info)
    
    # Define model loading function - FIXED: parameter name changed from model_args to args
    def build_model_fn(args):
        model = pretrained_model.s2t_model
        model.train()
        print(f'Trainable parameters: {count_parameters(model)}')
        return model
    
    # Set up experiment directories
    if args.output_dir is None:
        output_dir = f"/data/user_data/mkapadni/speech_project/exp/{args.language}/finetune"
    else:
        output_dir = args.output_dir
    
    if args.stats_dir is None:
        stats_dir = f"/data/user_data/mkapadni/speech_project/exp/{args.language}/stats_finetune"
    else:
        stats_dir = args.stats_dir
    
    # Ensure directories exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(stats_dir, exist_ok=True)
    
    # Update finetune config
    finetune_config = ez.config.update_finetune_config(
        's2t',
        pretrain_config,
        args.config_path
    )
    
    # Customize config with requested hyperparameters
    finetune_config['max_epoch'] = args.epochs
    finetune_config['num_iters_per_epoch'] = args.iterations
    finetune_config['batch_size'] = args.batch_size
    finetune_config['accum_grad'] = args.grad_accum
    
    # Update finetune config to ensure consistent data types
    finetune_config['dtype'] = 'float32'
    finetune_config['token_type'] = 'int64'
    # finetune_config['use_wandb'] = "true" if args.use_wandb else "false"
    finetune_config['use_wandb'] = "false"
    
    # Create trainer
    trainer = ez.Trainer(
        task='s2t',
        train_config=finetune_config,
        train_dataset=train_ez_dataset,
        valid_dataset=test_ez_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=output_dir,
        stats_dir=stats_dir,
        ngpu=args.ngpu
    )
    
    # Collect stats and train
    print("Collecting stats...")
    trainer.collect_stats()
    
    print(f"Starting training for {args.language}...")
    trainer.train()
    
    print(f"Training completed for {args.language}")

if __name__ == "__main__":
    main()