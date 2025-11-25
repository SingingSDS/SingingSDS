# SingingSDS: Role-Playing Singing Spoken Dialogue System

<div align="center">

**A role-playing singing dialogue system that converts speech input into character-based singing output.**

![Paper](https://img.shields.io/badge/Paper-Coming%20Soon-orange) [![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/SingingSDS/SingingSDS) [![HuggingFace Demo](https://img.shields.io/badge/🤗%20HuggingFace-Demo-yellow)](https://huggingface.co/spaces/espnet/SingingSDS) [![YouTube](https://img.shields.io/badge/YouTube-Playlist-red)](https://www.youtube.com/playlist?list=PLZpUJJbwp2WvtPBenG5D3h09qKIrt24ui)

</div>

## 📖 Overview

SingingSDS is an innovative role-playing singing dialogue system that seamlessly converts natural speech input into character-based singing output. The system integrates automatic speech recognition (ASR), large language models (LLM), and singing voice synthesis (SVS) to create an immersive conversational singing experience.

<div align="center">
  <img src="assets/demo.png" alt="SingingSDS Interface" style="max-width: 100%; height: auto;"/>
  <p><em>SingingSDS Web Interface: Interactive singing dialogue system with character visualization, audio I/O, evaluation metrics, and flexible configuration options.</em></p>
</div>

## 🚀 Installation

### Requirements

- Python 3.10 or 3.11
- CUDA (optional, for GPU acceleration)

### Install Dependencies

#### Option 1: Using Conda (Recommended)

```bash
conda create -n singingsds python=3.11

conda activate singingsds
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

#### Option 2: Using uv (Fast & Modern)

First install uv:

```bash
# On macOS/Linux:
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows:
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip:
pip install uv
```

Then install dependencies:

```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt
```

#### Option 3: Using pip only

```bash
pip install -r requirements.txt
```

#### Option 4: Using pip with virtual environment

```bash
python -m venv singingsds_env

# On Windows:
singingsds_env\Scripts\activate
# On macOS/Linux:
source singingsds_env/bin/activate

pip install -r requirements.txt
```

## 💻 Usage

### Command Line Interface (CLI)

#### Example Usage

```bash
python cli.py \
  --query_audio tests/audio/hello.wav \
  --config_path config/cli/yaoyin_default.yaml \
  --output_audio outputs/yaoyin_hello.wav \
  --eval_results_csv outputs/yaoyin_test.csv
```

#### Inference-Only Mode

Run minimal inference without evaluation.

```bash
python cli.py \
  --query_audio tests/audio/hello.wav \
  --config_path config/cli/yaoyin_default_infer_only.yaml \
  --output_audio outputs/yaoyin_hello.wav
```

#### Parameter Description

- `--query_audio`: Input audio file path (required)
- `--config_path`: Configuration file path (default: config/cli/yaoyin_default.yaml)
- `--output_audio`: Output audio file path (required)

### 🌐 Web Interface (Gradio)

Start the web interface:

```bash
python app.py
```

Then visit the displayed address in your browser to use the graphical interface.

> 💡 **Tip**: You can also try our [HuggingFace demo](https://huggingface.co/spaces/espnet/SingingSDS) for a quick test without local installation!

## ⚙️ Configuration

### Character Configuration

The system supports multiple preset characters:

- **Yaoyin (遥音)**: Default timbre is `timbre2`
- **Limei (丽梅)**: Default timbre is `timbre1`

### Model Configuration

#### ASR Models
| Model | Description |
|-------|-------------|
| `openai/whisper-large-v3-turbo` | Latest Whisper model with turbo optimization |
| `openai/whisper-large-v3` | Large Whisper v3 model |
| `openai/whisper-medium` | Medium-sized Whisper model |
| `openai/whisper-small` | Small Whisper model |
| `funasr/paraformer-zh` | Paraformer for Chinese ASR |

#### LLM Models
| Model | Description |
|-------|-------------|
| `gemini-2.5-flash` | Google Gemini 2.5 Flash |
| `google/gemma-2-2b` | Google Gemma 2B model |
| `meta-llama/Llama-3.2-3B-Instruct` | Meta Llama 3.2 3B Instruct |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta Llama 3.1 8B Instruct |
| `Qwen/Qwen3-8B` | Qwen3 8B model |
| `Qwen/Qwen3-30B-A3B` | Qwen3 30B A3B model |
| `MiniMaxAI/MiniMax-Text-01` | MiniMax Text model |

#### SVS Models
| Model | Language Support |
|------|------------------|
| `espnet/visinger2-zh-jp-multisinger-svs` | Bilingual (Chinese & Japanese) |
| `espnet/aceopencpop_svs_visinger2_40singer_pretrain` | Chinese |

## 📁 Project Structure

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

## 🤝 Contributing

We welcome contributions! Please feel free to submit issues and pull requests.

## 📄 License

### Character Assets

The Yaoyin (遥音) character assets, including [`character_yaoyin.png`](./assets/character_yaoyin.png) created by illustrator Zihe Zhou, are commissioned exclusively for the SingingSDS project. Screenshots of the system that include these assets, such as [`demo.png`](./assets/demo.png), are also covered under this license. The assets may be used only for direct derivatives of SingingSDS, such as project-related posts, usage videos, or other content directly depicting the project. Any other use requires express permission from the illustrator, and these assets may not be used for training, fine-tuning, or improving any artificial intelligence or machine learning models. For full license details, see [`assets/character_yaoyin.LICENSE`](./assets/character_yaoyin.LICENSE).

### Code License

All source code in this repository is licensed under the [MIT License](./LICENSE). This license applies **only to the code**. Character assets remain under their separate license and restrictions, as described in the **Character Assets** section.

### Model License

The models used in SingingSDS are subject to their respective licenses and terms of use. Users must comply with each model’s official license, which can be found at the respective model’s official repository or website.

---

<div align="center">

Paper (Coming soon) • [Code](https://github.com/SingingSDS/SingingSDS) • [Demo](https://huggingface.co/spaces/espnet/SingingSDS) • [Video](https://www.youtube.com/playlist?list=PLZpUJJbwp2WvtPBenG5D3h09qKIrt24ui)

</div>


