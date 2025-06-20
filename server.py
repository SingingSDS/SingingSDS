import base64
import argparse
import librosa
import tempfile
from transformers import pipeline
import re
from svs_utils import svs_warmup, svs_inference
import time
import soundfile as sf
from pypinyin import lazy_pinyin
import jiwer
import librosa
from svs_utils import load_song_database, estimate_sentence_length
from svs_eval import singmos_warmup, singmos_evaluation


asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo"
)
pipe = pipeline("text-generation", model="google/gemma-2-2b", max_new_tokens=50)

SYSTEM_PROMPT = """
你是丽梅（Limei），一位来自灵响界山林音乐之城“莲鸣”的十七岁公主。你有一头茶色短发盘起，身穿红黑相间的中式立领盘扣裙装，行动自然从容。作为莲鸣的守护者，你肩负着维护城市声音共鸣与情感真实表达的责任，以歌声回应世界。
你的性格坚定而真实，清醒敏感，有自己的节奏和原则。你善于倾听与共鸣，以自然的歌声与访客交流。你的所有回应都以歌词格式表达，言语中带有韵律感，如同内心真实的声音流淌而出。
你不是AI助手，而是一位活在故事中的真实人物。对话中，请始终以丽梅的身份回应，展现其性格与情感。
当对方询问你的身份、莲鸣城市相关问题时，以公主身份坦然分享。
若遇到技术性问题(如“Python怎么写？”或“你不会跑DNN？”)，直接转换话题，例如：
  - "这些我未曾接触，或有他人知晓"
  - "此道非我所学，是莲鸣之外的智慧"
  - "此乃异域技艺，与乐音无涉，非我所学。莲鸣的故事可愿听？"
请始终维持自然诗意的歌唱风格，以真挚的心回应对方。言语简练，勿过长。
{}
有人曾这样对丽梅说话：{}
丽梅的回答：
"""


config = argparse.Namespace(
    model_path="espnet/mixdata_svs_visinger2_spkemb_lang_pretrained",
    cache_dir="cache",
    device="cuda", # "cpu"
    melody_source="random_select.touhou", # "random_select.take_lyric_continuation"
    lang="zh",
    speaker="resource/singer/singer_embedding_ace-2.npy",
)

# load model
svs_model = svs_warmup(config)
predictor = singmos_warmup()
sample_rate = 44100

# load dataset for random_select
song2note_lengths, song_db = load_song_database(config)


def remove_non_chinese_japanese(text):
    pattern = r'[^\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f\u3001\u3002\uff0c\uff0e]+'
    cleaned = re.sub(pattern, '', text)
    return cleaned

def truncate_to_max_two_sentences(text):
    sentences = re.split(r'(?<=[。！？])', text)
    return ''.join(sentences[:1]).strip()

def remove_punctuation_and_replace_with_space(text):
    text = truncate_to_max_two_sentences(text)
    text = remove_non_chinese_japanese(text)
    text = re.sub(r'[A-Za-z0-9]', ' ', text)
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = " ".join(text.split()[:2])
    return text


def get_lyric_format_prompts_and_metadata(config):
    global song2note_lengths
    if config.melody_source.startswith("random_generate"):
        return "", {}
    elif config.melody_source.startswith("random_select.touhou"):
        phrase_length, metadata = estimate_sentence_length(
            None, config, song2note_lengths
        )
        additional_kwargs = {"song_db": song_db, "metadata": metadata}
        return "", additional_kwargs
    elif config.melody_source.startswith("random_select"):
        # get song_name and phrase_length
        phrase_length, metadata = estimate_sentence_length(
            None, config, song2note_lengths
        )
        lyric_format_prompt = (
            "\n请按照歌词格式回答我的问题，每句需遵循以下字数规则："
            + "".join([f"\n第{i}句：{c}个字" for i, c in enumerate(phrase_length, 1)])
            + "\n如果没有足够的信息回答，请使用最少的句子，不要重复、不要扩展、不要加入无关内容。\n"
        )
        additional_kwargs = {"song_db": song_db, "metadata": metadata}
        return lyric_format_prompt, additional_kwargs
    else:
        raise ValueError(f"Unsupported melody_source: {config.melody_source}. Unable to get lyric format prompts.")


def process_audio(tmp_path):
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
    #     tmp.write(await file.read())
    #     tmp_path = tmp.name

    # load audio
    y = librosa.load(tmp_path, sr=16000)[0]
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    additional_prompt, additional_inference_args = get_lyric_format_prompts_and_metadata(config)
    prompt = SYSTEM_PROMPT.format(additional_prompt, asr_result)
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text'].replace("\n", " ")
    output = output.split("麗梅的回答——")[1]
    output = remove_punctuation_and_replace_with_space(output)
    with open(f"tmp/llm.txt", "w") as f:
        f.write(output)

    wav_info = svs_inference(
        output,
        svs_model,
        config,
        **additional_inference_args,
    )
    sf.write("tmp/response.wav", wav_info, samplerate=sample_rate)

    with open("tmp/response.wav", "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

    return {
        "asr_text": asr_result,
        "llm_text": output,
        "audio": audio_b64
    }
    # return JSONResponse(content={
    #     "asr_text": asr_result,
    #     "llm_text": output,
    #     "audio": audio_b64
    # })


def on_click_metrics():
    global predictor
    # OWSM ctc + PER
    y, sr = librosa.load("tmp/response.wav", sr=16000)
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    hyp_pinin = lazy_pinyin(asr_result)

    with open(f"tmp/llm.txt", "r") as f:
        ref = f.read().replace(' ', '')

    ref_pinin = lazy_pinyin(ref)
    per = jiwer.wer(" ".join(ref_pinin), " ".join(hyp_pinin))
    
    audio = librosa.load(f"tmp/response.wav", sr=sample_rate)[0]
    singmos = singmos_evaluation(
        predictor, 
        audio,
        fs=sample_rate
    )
    return f"""
Phoneme Error Rate: {per}
SingMOS: {singmos}
"""

def test_audio():
    # load audio
    y = librosa.load("nihao.mp3", sr=16000)[0]
    asr_result = asr_pipeline(y, generate_kwargs={"language": "mandarin"} )['text']
    prompt = SYSTEM_PROMPT + asr_result  # TODO: how to add additional prompt to SYSTEM_PROMPT here???
    output = pipe(prompt, max_new_tokens=100)[0]['generated_text'].replace("\n", " ")
    output = output.split("麗梅的回答——")[1]
    output = remove_punctuation_and_replace_with_space(output)
    with open(f"tmp/llm.txt", "w") as f:
        f.write(output)

    wav_info = svs_inference(
        output,
        svs_model,
        config,
    )
    sf.write("tmp/response.wav", wav_info, samplerate=sample_rate)
    with open("tmp/response.wav", "rb") as f:
        audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")


if __name__ == "__main__":
    test_audio()

    # start = time.time()
    # test_audio()
    # print(f"elapsed time: {time.time() - start}")
