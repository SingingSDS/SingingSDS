import librosa
import numpy as np
import torch


def singmos_warmup():
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True
    )
    return predictor


def singmos_evaluation(predictor, wav_info, fs):
    wav_mos = librosa.resample(wav_info, orig_sr=fs, target_sr=16000)
    wav_mos = torch.from_numpy(wav_mos).unsqueeze(0)
    len_mos = torch.tensor([wav_mos.shape[1]])
    score = predictor(wav_mos, len_mos)
    return score


def score_extract_warmpup():
    from basic_pitch.inference import predict

    return predict


def score_metric_evaluation(score_extractor, audio_path):
    model_output, midi_data, note_events = score_extractor(audio_path)
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


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wav_path",
        type=Path,
        help="Path to the wav file",
    )
    parser.add_argument(
        "--results_csv",
        type=Path,
        help="csv file to save the results",
    )
    parser.parse_args()

    args = parser.parse_args()

    args.results_csv.parent.mkdir(parents=True, exist_ok=True)

    y, fs = librosa.load(args.wav_path, sr=None)

    # warmup
    predictor = singmos_warmup()
    score_extractor = score_extract_warmpup()

    # evaluate the audio
    metrics = {}

    # singmos evaluation
    score = singmos_evaluation(predictor, y, fs)
    metrics["singmos"] = score
    
    # score metric evaluation
    score_results = score_metric_evaluation(score_extractor, args.wav_path)
    metrics.update(score_results)

    # save results
    with open(args.results_csv, "a") as f:
        header = "file," + ",".join(metrics.keys()) + "\n"
        if f.tell() == 0:
            f.write(header)
        else:
            with open(args.results_csv, "r") as f2:
                file_header = f2.readline()
            if file_header != header:
                raise ValueError(f"Header mismatch: {file_header} vs {header}")

        line = (
            ",".join([str(args.wav_path)] + [str(v) for v in metrics.values()]) + "\n"
        )
        f.write(line)
