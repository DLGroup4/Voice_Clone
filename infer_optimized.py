# infer_batch_dynamic.py
import sys
import os
sys.path.append(os.path.abspath("fairseq-latest"))

import torch
import fairseq
import soundfile
import torch.nn.functional as F
import torchaudio.sox_effects as ta_sox
from typing import List

def transcribe_batch(model_path: str, audio_paths: List[str], device_id: int):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([model_path])
    model = model[0]
    model.to(device)
    model.eval()

    token = task.target_dictionary
    results = []

    for audio_path in audio_paths:
        audio, rate = soundfile.read(audio_path, dtype="float32")
        effects = [["gain", "-n"]]
        input_sample, rate = ta_sox.apply_effects_tensor(torch.tensor(audio).unsqueeze(0), rate, effects)
        input_sample = input_sample.float().to(device)

        with torch.no_grad():
            input_sample = F.layer_norm(input_sample, input_sample.shape)
            logits = model(source=input_sample, padding_mask=None)["encoder_out"]
            predicted_ids = torch.argmax(logits[:, 0], axis=-1)
            predicted_ids = torch.unique_consecutive(predicted_ids).tolist()
            transcription = token.string(predicted_ids)
            transcription = transcription.replace(' ', '').replace('|', ' ').strip()

        results.append({
            "filename": os.path.basename(audio_path),
            "transcript": transcription
        })

        # Clear memory per file to be safe
        del input_sample
        torch.cuda.empty_cache()

    # Full cleanup
    del model
    torch.cuda.empty_cache()

    return results

# === Called by subprocess
if __name__ == "__main__":
    model_path = sys.argv[1]
    device_id = int(sys.argv[2])
    audio_paths = sys.argv[3:]
    output = transcribe_batch(model_path, audio_paths, device_id)
    for res in output:
        print(res)

