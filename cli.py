from argparse import ArgumentParser
from logging import getLogger

import soundfile as sf
import yaml

from characters import CHARACTERS
from pipeline import SingingDialoguePipeline

logger = getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--query_audio", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="config/cli/yaoyin_default.yaml")
    parser.add_argument("--output_audio", type=str, required=True)
    return parser


def load_config(config_path: str):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = get_parser()
    args = parser.parse_args()
    config = load_config(args.config_path)
    pipeline = SingingDialoguePipeline(config)
    speaker = config["speaker"]
    language = config["language"]
    character_name = config["prompt_template_character"]
    character = CHARACTERS[character_name]
    prompt_template = character.prompt
    results = pipeline.run(args.query_audio, language, prompt_template, speaker)
    logger.info(
        f"Input: {args.query_audio}, Output: {args.output_audio}, ASR results: {results['asr_text']}, LLM results: {results['llm_text']}"
    )
    svs_audio, svs_sample_rate = results["svs_audio"]
    sf.write(args.output_audio, svs_audio, svs_sample_rate)


if __name__ == "__main__":
    main()
