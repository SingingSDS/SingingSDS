from modules.asr import get_asr_model
import librosa

if __name__ == "__main__":
    supported_asrs = [
        "funasr/paraformer-zh",
        "openai/whisper-large-v3-turbo",
    ]
    for model_id in supported_asrs:
        try:
            print(f"Loading model: {model_id}")
            asr_model = get_asr_model(model_id, device="auto", cache_dir=".cache")
            audio, sample_rate = librosa.load("tests/audio/hello.wav", sr=None)
            result = asr_model.transcribe(audio, sample_rate, language="mandarin")
            print(result)
        except Exception as e:
            print(f"Failed to load model {model_id}: {e}")
            breakpoint()
            continue