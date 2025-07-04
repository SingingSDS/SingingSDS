import librosa
import soundfile as sf
import numpy as np
import torch
import uuid
from pathlib import Path

# ----------- Initialization -----------


def init_singmos():
    print("[Init] Loading SingMOS...")
    return torch.hub.load(
        "South-Twilight/SingMOS:v0.3.0", "singing_ssl_mos", trust_repo=True
    )


def init_basic_pitch():
    print("[Init] Loading BasicPitch...")
    from basic_pitch.inference import predict

    return predict


def init_per():
    print("[Init] Loading PER...")
    from transformers import pipeline
    import jiwer

    asr_pipeline = pipeline(
        "automatic-speech-recognition", model="openai/whisper-large-v3-turbo"
    )
    return {
        "asr_pipeline": asr_pipeline,
        "jiwer": jiwer,
    }


def init_audiobox_aesthetics():
    print("[Init] Loading AudioboxAesthetics...")
    from audiobox_aesthetics.infer import initialize_predictor

    predictor = initialize_predictor()
    return predictor


# ----------- Evaluation -----------


def eval_singmos(audio_path, predictor):
    audio_array, sr = librosa.load(audio_path, sr=44100)
    wav = librosa.resample(audio_array, orig_sr=sr, target_sr=16000)
    wav_tensor = torch.from_numpy(wav).unsqueeze(0)
    length_tensor = torch.tensor([wav_tensor.shape[1]])
    score = predictor(wav_tensor, length_tensor)
    return {"singmos": float(score)}


def eval_melody_metrics(audio_path, pitch_extractor):
    model_output, midi_data, note_events = pitch_extractor(audio_path)
    metrics = {}
    assert (
        len(midi_data.instruments) == 1
    ), f"Detected {len(midi_data.instruments)} instruments for {audio_path}"
    midi_notes = midi_data.instruments[0].notes
    melody = [note.pitch for note in midi_notes]
    if len(melody) == 0:
        print(f"No notes detected in {audio_path}")
        return {}
    intervals = [abs(melody[i + 1] - melody[i]) for i in range(len(melody) - 1)]
    metrics["pitch_range"] = max(melody) - min(melody)
    if len(intervals) > 0:
        metrics["interval_mean"] = np.mean(intervals)
        metrics["interval_std"] = np.std(intervals)
        metrics["interval_large_jump_ratio"] = np.mean([i > 5 for i in intervals])
        metrics["dissonance_rate"] = compute_dissonance_rate(intervals)
    return metrics


def compute_dissonance_rate(intervals, dissonant_intervals={1, 2, 6, 10, 11}):
    dissonant = [i % 12 in dissonant_intervals for i in intervals]
    return np.mean(dissonant) if intervals else np.nan


def pypinyin_g2p_phone_without_prosody(text):
    from pypinyin import Style, pinyin
    from pypinyin.style._utils import get_finals, get_initials

    phones = []
    for phone in pinyin(text, style=Style.NORMAL, strict=False):
        initial = get_initials(phone[0], strict=False)
        final = get_finals(phone[0], strict=False)
        if len(initial) != 0:
            if initial in ["x", "y", "j", "q"]:
                if final == "un":
                    final = "vn"
                elif final == "uan":
                    final = "van"
                elif final == "u":
                    final = "v"
            if final == "ue":
                final = "ve"
            phones.append(initial)
            phones.append(final)
        else:
            phones.append(final)
    return phones


def eval_per(audio_path, reference_text, evaluator):
    audio_array, sr = librosa.load(audio_path, sr=16000)
    asr_result = evaluator["asr_pipeline"](
        audio_array, generate_kwargs={"language": "mandarin"}
    )["text"]
    hyp_pinyin = pypinyin_g2p_phone_without_prosody(asr_result)
    ref_pinyin = pypinyin_g2p_phone_without_prosody(reference_text)
    per = evaluator["jiwer"].wer(" ".join(ref_pinyin), " ".join(hyp_pinyin))
    return {"per": per}


def eval_aesthetic(audio_path, predictor):
    score = predictor.forward([{"path": str(audio_path)}])
    return score


# ----------- Main Function -----------


def load_evaluators(config):
    loaded = {}
    if "singmos" in config:
        loaded["singmos"] = init_singmos()
    if "melody" in config:
        loaded["melody"] = init_basic_pitch()
    if "per" in config:
        loaded["per"] = init_per()
    if "aesthetic" in config:
        loaded["aesthetic"] = init_audiobox_aesthetics()
    return loaded


def run_evaluation(audio_path, evaluators, **kwargs):
    results = {}
    if "singmos" in evaluators:
        results.update(eval_singmos(audio_path, evaluators["singmos"]))
    if "per" in evaluators:
        results.update(eval_per(audio_path, kwargs["llm_text"], evaluators["per"]))
    if "melody" in evaluators:
        results.update(eval_melody_metrics(audio_path, evaluators["melody"]))
    if "aesthetic" in evaluators:
        results.update(eval_aesthetic(audio_path, evaluators["aesthetic"])[0])
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--wav_path", type=str, required=True)
    parser.add_argument("--results_csv", type=str, required=True)
    parser.add_argument("--evaluators", type=str, default="singmos,melody,aesthetic")
    args = parser.parse_args()
    evaluators = load_evaluators(args.evaluators.split(","))
    results = run_evaluation(args.wav_path, evaluators)
    print(results)

    with open(args.results_csv, "a") as f:
        header = "file," + ",".join(results.keys()) + "\n"
        if f.tell() == 0:
            f.write(header)
        else:
            with open(args.results_csv, "r") as f2:
                file_header = f2.readline()
            if file_header != header:
                raise ValueError(f"Header mismatch: {file_header} vs {header}")
        line = (
            ",".join([str(args.wav_path)] + [str(v) for v in results.values()]) + "\n"
        )
        f.write(line)
