---
title: SingingSDS
emoji: 🎶
colorFrom: pink
colorTo: yellow
sdk: gradio
sdk_version: 5.4.0
app_file: app.py
pinned: false
python_version: 3.11
---
# SingingSDS: Role-Playing Singing Spoken Dialogue System

A role-playing singing dialogue system that converts speech input into character-based singing output.

## Installation

### Requirements

- Python 3.11+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

#### Option 1: Using Conda (Recommended)

```bash
conda create -n singingsds python=3.11

conda activate singingsds
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Option 2: Using pip only

```bash
pip install -r requirements.txt
```

#### Option 3: Using pip with virtual environment

```bash
python -m venv singingsds_env

# On Windows:
singingsds_env\Scripts\activate
# On macOS/Linux:
source singingsds_env/bin/activate

pip install -r requirements.txt
```

## Usage

### Command Line Interface (CLI)

#### Example Usage

```bash
python cli.py --query_audio tests/audio/hello.wav --config_path config/cli/yaoyin_default.yaml --output_audio outputs/yaoyin_hello.wav
```

#### Parameter Description

- `--query_audio`: Input audio file path (required)
- `--config_path`: Configuration file path (default: config/cli/yaoyin_default.yaml)
- `--output_audio`: Output audio file path (required)


### Web Interface (Gradio)

Start the web interface:

```bash
python app.py
```

Then visit the displayed address in your browser to use the graphical interface.

## Configuration

### Character Configuration

The system supports multiple preset characters:

- **Yaoyin (遥音)**: Default timbre is `timbre2`
- **Limei (丽梅)**: Default timbre is `timbre1`

### Model Configuration

#### ASR Models
- `openai/whisper-large-v3-turbo`
- `openai/whisper-large-v3`
- `openai/whisper-medium`
- `sanchit-gandhi/whisper-small-dv`
- `facebook/wav2vec2-base-960h`

#### LLM Models
- `google/gemma-2-2b`
- `MiniMaxAI/MiniMax-M1-80k`
- `meta-llama/Llama-3.2-3B-Instruct`

#### SVS Models
- `espnet/mixdata_svs_visinger2_spkemb_lang_pretrained_avg` (Bilingual)
- `espnet/aceopencpop_svs_visinger2_40singer_pretrain` (Chinese)

## Project Structure

```
SingingSDS/
├── app.py, cli.py               # Entry points (demo app & CLI)
├── pipeline.py                  # Main orchestration pipeline
├── interface.py                 # Gradio interface
├── characters/                  # Virtual character definitions
├── modules/                     # Core modules
│   ├── asr/                     # ASR models (Whisper, Paraformer)
│   ├── llm/                     # LLMs (Gemini, LLaMA, etc.)
│   ├── svs/                     # Singing voice synthesis (ESPnet)
│   └── utils/                   # G2P, text normalization, resources
├── config/                      # YAML configuration files 
├── data/                        # Dataset metadata and length info
├── data_handlers/               # Parsers for KiSing, Touhou, etc.
├── evaluation/                  # Evaluation metrics
├── resources/                   # Singer embeddings, phoneme dicts, MIDI
├── assets/                      # Character visuals
├── tests/                       # Unit tests and sample audios
└── README.md, requirements.txt
```

## Contributing

Issues and Pull Requests are welcome!

## License


