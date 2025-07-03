from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

import yaml

from characters import get_character
from pipeline import SingingDialoguePipeline

logger = getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--query_audio", type=Path, required=True)
    parser.add_argument(
        "--config_path", type=Path, default="config/cli/yaoyin_default.yaml"
    )
    parser.add_argument("--output_audio", type=Path, required=True)
    return parser


def load_config(config_path: Path):
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
    character = get_character(character_name)
    prompt_template = character.prompt
    results = pipeline.run(
        args.query_audio,
        language,
        prompt_template,
        speaker,
        output_audio_path=args.output_audio,
    )
    logger.info(
        f"Input: {args.query_audio}, Output: {args.output_audio}, ASR results: {results['asr_text']}, LLM results: {results['llm_text']}"
    )


if __name__ == "__main__":
    main()
