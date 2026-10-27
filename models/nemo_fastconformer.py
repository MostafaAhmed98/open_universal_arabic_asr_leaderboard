import json
import time
from pathlib import Path

import torch
from tqdm import tqdm
from nemo.collections.asr.models import ASRModel

device = "cuda:0"


def run_nemo_fastconformer(model_path, data_manifest, data_folder, output_manifest, batch_size=8):
    """
    Arguments
    ---------
    model_path: str
        Local path to the .nemo checkpoint of the FastConformer model
    data_manifest: str
        Path of a data manifest under datasets/
    data_folder: str
        The path to the test set
    output_manifest: str
        The output manifest path
    batch_size: int
        Number of clips decoded together

    Output
    ---------
    Create an output manifest containing ground truths and predictions
    """
    model_path = Path(model_path).resolve()
    model = ASRModel.restore_from(restore_path=str(model_path), map_location=device)
    model.eval()
    model = model.to(device)

    entries = []
    with open(data_manifest, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    audio_paths = [
        str(Path(e["audio_filepath"].format(data_folder=data_folder)).resolve()) for e in entries
    ]

    torch.cuda.reset_peak_memory_stats(torch.device(device))
    initial_memory = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 3)
    all_inference_time = 0
    all_audio_duration = 0
    hypotheses = []

    for count, start in enumerate(tqdm(range(0, len(audio_paths), batch_size))):
        batch_paths = audio_paths[start:start + batch_size]
        batch_entries = entries[start:start + batch_size]

        torch.cuda.synchronize(torch.device(device))
        start_time = time.time()
        with torch.inference_mode():
            batch_hyps = model.transcribe(audio=batch_paths, batch_size=batch_size, return_hypotheses=False)
        if isinstance(batch_hyps, tuple):
            batch_hyps = batch_hyps[0]
        torch.cuda.synchronize(torch.device(device))
        end_time = time.time()

        if len(batch_hyps) != len(batch_paths):
            raise RuntimeError(
                f"NeMo returned {len(batch_hyps)} hypotheses for {len(batch_paths)} audio files"
            )
        if count > 4:
            all_inference_time += end_time - start_time
            all_audio_duration += sum(e["duration"] for e in batch_entries)
        hypotheses.extend(batch_hyps)
    peak_memory = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 ** 3)

    with open(output_manifest, "w", encoding="utf-8") as fout:
        for entry, path, pred in zip(entries, audio_paths, hypotheses):
            pred_text = pred.text if hasattr(pred, "text") else str(pred)
            metadata = {
                "audio_filepath": path,
                "text": entry["text"],
                "pred_text": pred_text,
            }
            json.dump(metadata, fout, ensure_ascii=False)
            fout.write("\n")

    print("average rtf : ", all_inference_time / all_audio_duration if all_audio_duration else "n/a")
    print("model memory : ", initial_memory)
    print("peak inference memory : ", peak_memory - initial_memory)