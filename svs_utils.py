from util import (
    preprocess_input,
    postprocess_phn,
    get_tokenizer,
    get_pinyin,
)
from espnet_model_zoo.downloader import ModelDownloader
from espnet2.bin.svs_inference import SingingGenerate
import librosa
import torch
import numpy as np
import random
import json

import argparse
import soundfile as sf

# the code below should be in app.py than svs_utils.py
# espnet_model_dict = {
#     "Model①(Chinese)-zh": "espnet/aceopencpop_svs_visinger2_40singer_pretrain",
#     "Model②(Multilingual)-zh": "espnet/mixdata_svs_visinger2_spkembed_lang_pretrained",
#     "Model②(Multilingual)-jp": "espnet/mixdata_svs_visinger2_spkembed_lang_pretrained",
# }


singer_embeddings = {
    "espnet/aceopencpop_svs_visinger2_40singer_pretrain": {
        "singer1 (male)": 1,
        "singer2 (female)": 12,
        "singer3 (male)": 23,
        "singer4 (female)": 29,
        "singer5 (male)": 18,
        "singer6 (female)": 8,
        "singer7 (male)": 25,
        "singer8 (female)": 5,
        "singer9 (male)": 10,
        "singer10 (female)": 15,
    },
    "espnet/mixdata_svs_visinger2_spkembed_lang_pretrained": {
        "singer1 (male)": "resource/singer/singer_embedding_ace-1.npy",
        "singer2 (female)": "resource/singer/singer_embedding_ace-2.npy",
        "singer3 (male)": "resource/singer/singer_embedding_ace-3.npy",
        "singer4 (female)": "resource/singer/singer_embedding_ace-8.npy",
        "singer5 (male)": "resource/singer/singer_embedding_ace-7.npy",
        "singer6 (female)": "resource/singer/singer_embedding_itako.npy",
        "singer7 (male)": "resource/singer/singer_embedding_ofuton.npy",
        "singer8 (female)": "resource/singer/singer_embedding_kising_orange.npy",
        "singer9 (male)": "resource/singer/singer_embedding_m4singer_Tenor-1.npy",
        "singer10 (female)": "resource/singer/singer_embedding_m4singer_Alto-4.npy",
    },
}


def svs_warmup(config):
    """
    What: module loading, and model loading
    Input: config dict/namespace (e.g., model path, cache dir, device, language, possibly speaker selection)
    Return: the inference prototype function (which creates pitch/duration and runs model-specific inference)
    """
    if config.model_path.startswith("espnet"):
        espnet_downloader = ModelDownloader(config.cache_dir)
        downloaded = espnet_downloader.download_and_unpack(config.model_path)
        model = SingingGenerate(
            train_config=downloaded["train_config"],
            model_file=downloaded["model_file"],
            device=config.device,
        )
    else:
        raise NotImplementedError(f"Model {config.model_path} not supported")
    return model


def svs_text_preprocessor(model_path, texts, lang):
    """
    Input:
        - model_path (str), for getting the corresponding tokenizer
        - texts (str), in Chinese character or Japanese character
        - lang (str), language label jp/zh, input if is not espnet model

    Output:
        - lyric_ls (lyric list), each element as 'k@zhe@zh'
        - sybs (phn w/ _ list), each element as 'k@zh_e@zh'
        - labels (phn w/o _ list), each element as 'k@zh'

    """
    fs = 44100

    if texts is None:
        return (fs, np.array([0.0])), "Error: No Text provided!"

    # preprocess
    if lang == "zh":
        texts = preprocess_input(texts, "")
        text_list = get_pinyin(texts)
    elif lang == "jp":
        texts = preprocess_input(texts, " ")
        text_list = texts.strip().split()

    # text to phoneme
    tokenizer = get_tokenizer(model_path, lang)
    sybs = []  # phoneme list
    for text in text_list:
        if text == "AP" or text == "SP":
            rev = [text]
        elif text == "-" or text == "——":
            rev = [text]
        else:
            rev = tokenizer(text)
        if rev == False:
            return (fs, np.array([0.0])), f"Error: text `{text}` is invalid!"
        rev = postprocess_phn(rev, model_path, lang)
        phns = "_".join(rev)
        sybs.append(phns)

    lyric_ls = []
    labels = []
    pre_phn = ""
    for phns in sybs:
        if phns == "-" or phns == "——":
            phns = pre_phn

        phn_list = phns.split("_")
        lyric = "".join(phn_list)
        for phn in phn_list:
            labels.append(phn)
        pre_phn = labels[-1]
        lyric_ls.append(lyric)

    return lyric_ls, sybs, labels


def svs_get_batch(model_path, answer_text, lang, random_gen=True):
    """
    Input:
        - answer_text (str), in Chinese character or Japanese character
        - model_path (str), loaded pretrained model name
        - lang (str), language label jp/zh, input if is not espnet model
    Output:
        - batch (dict)

    {'score': (75, [[0, 0.48057527844210024, 'n@zhi@zh', 66, 'n@zh_i@zh'],
            [0.48057527844210024, 0.8049310140914353, 'k@zhe@zh', 57, 'k@zh_e@zh'],
            [0.8049310140914353, 1.1905956333296641, 'm@zhei@zh', 64, 'm@zh_ei@zh']]),
     'text': 'n@zh i@zh k@zh e@zh m@zh ei@zh'}
    """
    tempo = 120
    lyric_ls, sybs, labels = svs_text_preprocessor(model_path, answer_text, lang)
    len_note = len(lyric_ls)
    notes = []
    if random_gen:
        # midi_range = (57,69)
        st = 0
        for id_lyric in range(len_note):
            pitch = random.randint(57, 69)
            period = round(random.uniform(0.1, 0.5), 4)
            ed = st + period
            note = [st, ed, lyric_ls[id_lyric], pitch, sybs[id_lyric]]
            st = ed
            notes.append(note)

        phns_str = " ".join(labels)
        batch = {
            "score": (
                int(tempo),
                notes,
            ),
            "text": phns_str,
        }

    # print(batch)
    return batch


langs = {
    "zh": 2,
    "jp": 1,
    "en": 2,
}

exist_model = "Null"
svs = None


def svs_inference(model_name, model_svs, answer_text, lang, random_gen=True, fs=44100):
    batch = svs_get_batch(model_name, answer_text, lang, random_gen=random_gen)

    # Infer
    spk = "singer1 (male)"
    global exist_model
    global svs
    svs = model_svs
    exist_model = model_name
    # if exist_model == "Null" or exist_model != model_name:
    #     # device = "cpu"
    #     device = "cuda" if torch.cuda.is_available() else "cpu"
    #     d = ModelDownloader(cachedir="./cache")
    #     pretrain_downloaded = d.download_and_unpack(model_name)
    #     svs = SingingGenerate(
    #         train_config = pretrain_downloaded["train_config"],
    #         model_file = pretrain_downloaded["model_file"],
    #         device = device
    #     )
    #     exist_model = model_name
    if model_name == "Model①(Chinese)-zh":
        sid = np.array([singer_embeddings[model_name][spk]])
        output_dict = svs(batch, sids=sid)
    else:
        lid = np.array([langs[lang]])
        spk_embed = np.load("resource/singer/singer_embedding_ace-1.npy")
        output_dict = svs(batch, lids=lid, spembs=spk_embed)
    wav_info = output_dict["wav"].cpu().numpy()

    return wav_info


def singmos_warmup(config):
    predictor = torch.hub.load(
        "South-Twilight/SingMOS:v0.2.0", "singing_ssl_mos", trust_repo=True
    )
    return predictor, "South-Twilight/SingMOS:v0.2.0"


def singmos_evaluation(predictor, wav_info, fs):
    wav_mos = librosa.resample(wav_info, orig_sr=fs, target_sr=16000)
    wav_mos = torch.from_numpy(wav_mos).unsqueeze(0)
    len_mos = torch.tensor([wav_mos.shape[1]])
    score = predictor(wav_mos, len_mos)
    return score


def estimate_sentence_length(query, config, song2note_lengths):
    if config.melody_source.startswith("random_select"):
        # random select a song from database, and return its value in the phrase_length column
        # return phrase_length column and song name
        song_name = random.choice(list(song2note_lengths.keys()))
        phrase_length = song2note_lengths[song_name]
        metadata = {"song_name": song_name}
        return phrase_length, metadata
    else:
        raise NotImplementedError(f"melody source {config.melody_source} not supported")


def align_score_and_text(segment_iterator, lyric_ls, sybs, labels, config):
    text = []
    lyric_idx = 0
    notes_info = []
    while lyric_idx < len(lyric_ls):
        score = next(segment_iterator)
        for note_start_time, note_end_time, reference_note_lyric, note_midi in zip(
            score["note_start_times"],
            score["note_end_times"],
            score["note_lyrics"],
            score["note_midi"],
        ):
            if reference_note_lyric in ["<AP>", "<SP>"]:
                notes_info.append(
                    [
                        note_start_time,
                        note_end_time,
                        reference_note_lyric.strip("<>"),
                        note_midi,
                        reference_note_lyric.strip("<>"),
                    ]
                )
                text.append(reference_note_lyric.strip("<>"))
            elif reference_note_lyric in ["-", "——"] and config.melody_source == "random_select.take_lyric_continuation":
                notes_info.append(
                    [
                        note_start_time,
                        note_end_time,
                        reference_note_lyric,
                        note_midi,
                        text[-1],
                    ]
                )
            else:
                notes_info.append(
                    [
                        note_start_time,
                        note_end_time,
                        lyric_ls[lyric_idx],
                        note_midi,
                        sybs[lyric_idx],
                    ]
                )
                text += sybs[lyric_idx].split("_")
                lyric_idx += 1
                if lyric_idx >= len(lyric_ls):
                    break
    batch = {
        "score": (
            score["tempo"],  # Assume the tempo is the same for all segments
            notes_info,
        ),
        "text": " ".join(text),
    }
    return batch


def song_segment_iterator(song_db, metadata):
    song_name = metadata["song_name"]
    if song_name.startswith("kising_"):
        # return a iterator that load from song_name_{001} and increment
        segment_id = 1
        while f"{song_name}_{segment_id:03d}" in song_db.index:
            yield song_db.loc[f"{song_name}_{segment_id:03d}"]
            segment_id += 1
    else:
        raise NotImplementedError(f"song name {song_name} not supported")


def load_song_database():
    song_db = load_dataset(
        "jhansss/kising_score_segments", cache_dir="cache", split="train"
    ).to_pandas()
    song_db.set_index("segment_id", inplace=True)

    with open("data/song2note_lengths.json", "r") as f:
        song2note_lengths = json.load(f)
    return song2note_lengths, song_db


if __name__ == "__main__":

    # -------- demo code for generate audio from randomly selected song ---------#
    config = argparse.Namespace(
        model_path="espnet/mixdata_svs_visinger2_spkembed_lang_pretrained",
        cache_dir="cache",
        device="cuda", # "cpu"
        melody_source="random_generate", # "random_select.take_lyric_continuation"
        lang="zh",
    )

    # load model
    model = svs_warmup(config)

    answer_text = "天气真好\n空气清新\n气温温和\n风和日丽\n天高气爽\n阳光明媚"

    sample_rate = 44100

    if config.melody_source.startswith("random_select"):
        # load song database: jhansss/kising_score_segments
        from datasets import load_dataset
        song2note_lengths, song_db = load_song_database()

        # get song_name and phrase_length
        phrase_length, metadata = estimate_sentence_length(None, config, song2note_lengths)

        # then, phrase_length info should be added to llm prompt, and get the answer lyrics from llm
        # e.g. answer_text = "天气真好\n空气清新"
        lyric_ls, sybs, labels = svs_text_preprocessor(
            config.model_path, answer_text, config.lang
        )
        segment_iterator = song_segment_iterator(song_db, metadata)
        batch = align_score_and_text(segment_iterator, lyric_ls, sybs, labels, config)
        singer_embedding = np.load(singer_embeddings[config.model_path]["singer2 (female)"])
        lid = np.array([langs[config.lang]])
        output_dict = model(batch, lids=lid, spembs=singer_embedding)
        wav_info = output_dict["wav"].cpu().numpy()

    
    elif config.melody_source.startswith("random_generate"):
        wav_info = svs_inference(config.model_path, model, answer_text, lang=config.lang, random_gen=True, fs=sample_rate)

    # write wav to output_retrieved.wav
    save_name = config.melody_source.split('.')[0]
    sf.write(f"{save_name}.wav", wav_info, samplerate=sample_rate)
