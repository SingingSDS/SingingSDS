from argparse import ArgumentParser
from logging import getLogger
from pathlib import Path

import yaml

from characters import get_character
from pipeline import SingingDialoguePipeline

logger = getLogger(__name__)


def get_parser():
    parser = ArgumentParser()
    parser.add_argument("--query_audios", nargs="+", type=Path, required=True)
    parser.add_argument(
        "--config_path", type=Path, default="config/cli/yaoyin_default.yaml"
    )
    parser.add_argument("--output_audio_folder", type=Path, required=True)
    parser.add_argument("--eval_results_csv", type=Path, required=True)
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
    args.output_audio_folder.mkdir(parents=True, exist_ok=True)
    args.eval_results_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.eval_results_csv, "a") as f:
        f.write(
            f"query_audio,asr_model,llm_model,svs_model,melody_source,language,speaker,output_audio,asr_text,llm_text,metrics\n"
        )
    try:
        for query_audio in args.query_audios:
            output_audio = args.output_audio_folder / f"{query_audio.stem}_response.wav"
            results = pipeline.run(
                query_audio,
                language,
                prompt_template,
                speaker,
                output_audio_path=output_audio,
            )
            metrics = pipeline.evaluate(output_audio, **results)
            metrics.update(results.get("metrics", {}))
            metrics_str = ",".join([f"{metrics[k]}" for k in sorted(metrics.keys())])
            logger.info(
                f"Input: {query_audio}, Output: {output_audio}, ASR results: {results['asr_text']}, LLM results: {results['llm_text']}"
            )
            with open(args.eval_results_csv, "a") as f:
                f.write(
                    f"{query_audio},{config['asr_model']},{config['llm_model']},{config['svs_model']},{config['melody_source']},{config['language']},{config['speaker']},{output_audio},{results['asr_text']},{results['llm_text']},{metrics_str}\n"
                )
    except Exception as e:
        logger.error(f"Error in main: {e}")
        breakpoint()
        raise e


if __name__ == "__main__":
    main()
