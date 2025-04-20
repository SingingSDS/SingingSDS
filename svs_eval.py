import librosa
import pyworld as pw
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


def pitch_interval_evaluation(y, fs):
    _f0, t = pw.dio(y.astype(np.float64), fs)
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, fs)

    f0[f0 == 0] = np.nan
    midi_f0 = librosa.hz_to_midi(f0)

    if len(midi_f0) < 2:
        return np.nan, np.nan

    # only consider the intervals between notes
    intervals = np.diff(midi_f0)
    intervals = intervals[~np.isnan(intervals)]
    interval_mean = np.mean(np.abs(intervals))
    interval_std = np.std(intervals)
    return interval_mean, interval_std


def chroma_entropy_evaluation(y, fs):
    chroma = librosa.feature.chroma_cqt(y=y, sr=fs)
    chroma_sum = np.sum(chroma, axis=0, keepdims=True)
    chroma_sum = np.clip(chroma_sum, 1e-6, None)
    chroma_norm = chroma / chroma_sum
    chroma_norm = np.clip(chroma_norm, 1e-6, 1.0)
    entropy = -np.sum(chroma_norm * np.log2(chroma_norm), axis=0)
    return np.mean(entropy)


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

    # singmos evaluation
    score = singmos_evaluation(predictor, y, fs)

    # pitch interval evaluation
    interval_mean, interval_std = pitch_interval_evaluation(y, fs)
    # chroma entropy evaluation
    chroma_entropy = chroma_entropy_evaluation(y, fs)
    
    # # visualize
    # import matplotlib.pyplot as plt
    # import librosa.display
    # chroma = librosa.feature.chroma_cqt(y=y, sr=fs)
    # img = librosa.display.specshow(chroma, y_axis='chroma', x_axis='time')
    # plt.colorbar(img)
    # plt.savefig(args.results_csv.parent / args.wav_path.with_suffix('.png'))

    # save results
    results = {
        "singmos": score,
        "pitch_interval_mean": interval_mean,
        "pitch_interval_std": interval_std,
        "chroma_entropy": chroma_entropy,
    }

    with open(args.results_csv, "a") as f:
        header = "file," + ",".join(results.keys()) + "\n"
        if f.tell() == 0:
            f.write(header)
        else:
            with open(args.results_csv, "r") as f2:
                file_header = f2.readline()
            if file_header != header:
                raise ValueError(
                    f"Header mismatch: {file_header} vs {header}"
                )

        line = ",".join([str(args.wav_path)] + [str(v) for v in results.values()]) + "\n"
        f.write(line)
