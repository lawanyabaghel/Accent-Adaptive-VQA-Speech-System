#!/usr/bin/env python3
import argparse
from pathlib import Path
from espnetez.trainer import Trainer
from espnet2.bin.tts_inference import Text2Speech
import soundfile as sf

def get_parser():
    parser = argparse.ArgumentParser(
        description="Finetune VITS (or other ESPnet2‐TTS) on Afrispeech and run a test synthesis"
    )
    parser.add_argument(
        "--model",
        default="kan-bayashi/ljspeech_vits",
        type=str,
        help="Pretrained TTS model tag or local checkpoint"
    )
    parser.add_argument(
        "--vocoder",
        default="parallel_wavegan/ljspeech_parallel_wavegan.v1",
        type=str,
        help="Vocoder tag or local checkpoint"
    )
    parser.add_argument(
        "--dump-dir",
        type=Path,
        default=Path("dump/afrispeech"),
        help="Root of your Afrispeech dump (must contain train/ and valid/ subdirs)"
    )
    parser.add_argument(
        "--stats-dir",
        type=Path,
        default=Path("dump/afrispeech/stats"),
        help="Where to write feature statistics"
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=Path("exp/finetune_afrispeech"),
        help="Where to save the finetuned model"
    )
    parser.add_argument(
        "--init-param",
        type=str,
        default=None,
        help="Optional: path to a .pth to load via --init_param"
    )
    parser.add_argument(
        "--test-text",
        type=str,
        default="This is a test of the finetuned Afrispeech model.",
        help="Text to synthesize after finetuning"
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=Path("afrispeech_test.wav"),
        help="Path to write the test WAV"
    )
    return parser

def main():
    args = get_parser().parse_args()

    # 1) Build train_config for Trainer
    train_config = {
        "model_file": args.model,
        "vocoder_file": args.vocoder,
        # these YAMLs assume VITS defaults; change for Tacotron2/FS2 if needed
        "train_config": "conf/tuning/train_vits.yaml",
        "inference_config": "conf/tuning/decode_vits.yaml",
    }
    if args.init_param:
        train_config["init_param"] = args.init_param

    # 2) Map your dumped files: "text" from text, "speech" from wav.scp
    data_info = {
        "text": ("text", "text"),
        "speech": ("wav.scp", "sound"),
    }

    # 3) Instantiate Trainer and run stats → train
    trainer = Trainer(
        task="tts",
        train_config=train_config,
        output_dir=str(args.exp_dir),
        stats_dir=str(args.stats_dir),
        data_info=data_info,
        train_dump_dir=str(args.dump_dir / "train"),
        valid_dump_dir=str(args.dump_dir / "valid"),
    )
    trainer.collect_stats()
    trainer.train()

    # 4) Quick inference test
    config_file = args.exp_dir / "config.yaml"
    model_file  = args.exp_dir / "latest.pth"
    # choose vocoder argument
    voc_kw = {"vocoder_tag": args.vocoder}
    if Path(args.vocoder).exists():
        voc_kw = {"vocoder_file": args.vocoder}

    tts = Text2Speech.from_pretrained(
        config_file=str(config_file),
        model_file=str(model_file),
        **voc_kw
    )
    wav = tts(args.test_text)["wav"]
    sf.write(str(args.test_output), wav, tts.fs)
    print(f"Test synthesis saved to {args.test_output}")

if __name__ == "__main__":
    main()
