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
python cli.py --query_audio data/query/hello.wav --config_path config/cli/yaoyin_default.yaml --output_audio outputs/yaoyin_hello.wav
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
- `espnet/mixdata_svs_visinger2_spkemb_lang_pretrained` (Bilingual)
- `espnet/aceopencpop_svs_visinger2_40singer_pretrain` (Chinese)

## Project Structure

```
SingingSDS/
├── cli.py                 # Command line interface
├── interface.py           # Gradio interface
├── pipeline.py            # Core processing pipeline
├── app.py                 # Web application entry
├── requirements.txt       # Python dependencies
├── config/                # Configuration files
│   ├── cli/               # CLI-specific configuration
│   └── interface/         # Interface-specific configuration
├── modules/               # Core modules
│   ├── asr.py            # Speech recognition module
│   ├── llm.py            # Large language model module
│   ├── melody.py         # Melody control module
│   ├── svs/              # Singing voice synthesis modules
│   │   ├── base.py       # Base SVS class
│   │   ├── espnet.py     # ESPnet SVS implementation
│   │   ├── registry.py   # SVS model registry
│   │   └── __init__.py   # SVS module initialization
│   └── utils/            # Utility modules
│       ├── g2p.py        # Grapheme-to-phoneme conversion
│       ├── text_normalize.py # Text normalization
│       └── resources/    # Utility resources
├── characters/            # Character definitions
│   ├── base.py           # Base character class
│   ├── Limei.py          # Limei character definition
│   ├── Yaoyin.py         # Yaoyin character definition
│   └── __init__.py       # Character module initialization
├── evaluation/            # Evaluation modules
│   └── svs_eval.py       # SVS evaluation metrics
├── data/                  # Data directory
│   ├── kising/           # Kising dataset
│   └── touhou/           # Touhou dataset
├── resources/             # Project resources
├── data_handlers/         # Data handling utilities
├── assets/                # Static assets
└── tests/                 # Test files
```

## Contributing

Issues and Pull Requests are welcome!

## License


