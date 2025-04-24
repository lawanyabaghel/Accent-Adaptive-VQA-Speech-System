from scipy.io.wavfile import write
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import torch
from huggingface_hub import snapshot_download

# Download the entire model repository (all files)
model_path = snapshot_download(repo_id="intronhealth/afro-tts")
print(f"Downloaded model to: {model_path}")

# Load the model using the downloaded path
config = XttsConfig()
config.load_json(f"{model_path}/config.json")
model = Xtts.init_from_config(config)

# Pass the model_path directly as the checkpoint_dir
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
model.cuda()

# Now synthesize speech
outputs = model.synthesize(
    "Abraham said today is a good day to sound like an African.",
    config,
    speaker_wav=f"{model_path}/audios/reference_accent.wav",
    gpt_cond_len=3,
    language="en",
)

write("output.wav", 24000, outputs['wav'])
