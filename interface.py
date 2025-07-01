import time
import uuid

import gradio as gr
import yaml

from characters import CHARACTERS
from pipeline import SingingDialoguePipeline


class GradioInterface:
    def __init__(self, options_config: str, default_config: str):
        self.options = self.load_config(options_config)
        self.svs_model_map = {
            model["id"]: model for model in self.options["svs_models"]
        }
        self.default_config = self.load_config(default_config)
        self.character_info = CHARACTERS
        self.current_character = self.default_config["character"]
        self.current_svs_model = (
            f"{self.default_config['language']}-{self.default_config['svs_model']}"
        )
        self.current_voice = self.svs_model_map[self.current_svs_model]["voices"][
            self.character_info[self.current_character].default_voice
        ]
        self.pipeline = SingingDialoguePipeline(self.default_config)

    def load_config(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def create_interface(self) -> gr.Blocks:
        try:
            with gr.Blocks(title="SingingSDS") as demo:
                gr.Markdown("# SingingSDS: Role-Playing Singing Spoken Dialogue System")
                with gr.Row():
                    with gr.Column(scale=1):
                        character_image = gr.Image(
                            self.character_info[self.current_character].image_path,
                            label="Character",
                            show_label=False,
                        )
                    with gr.Column(scale=2):
                        mic_input = gr.Audio(
                            sources=["microphone", "upload"],
                            type="filepath",
                            label="Speak to the character",
                        )
                        interaction_log = gr.Textbox(
                            label="Interaction Log", lines=3, interactive=False
                        )
                        audio_output = gr.Audio(
                            label="Character's Response", type="filepath", autoplay=True
                        )

                        with gr.Row():
                            metrics_button = gr.Button(
                                "Evaluate Metrics", variant="secondary"
                            )
                            metrics_output = gr.Textbox(
                                label="Evaluation Results", lines=3, interactive=False
                            )

                gr.Markdown("## Configuration")
                with gr.Row():
                    with gr.Column():
                        character_radio = gr.Radio(
                            label="Character Role",
                            choices=list(self.character_info.keys()),
                            value=self.default_config["character"],
                        )
                        with gr.Row():
                            asr_radio = gr.Radio(
                                label="ASR Model",
                                choices=[
                                    (model["name"], model["id"])
                                    for model in self.options["asr_models"]
                                ],
                                value=self.default_config["asr_model"],
                            )
                        with gr.Row():
                            llm_radio = gr.Radio(
                                label="LLM Model",
                                choices=[
                                    (model["name"], model["id"])
                                    for model in self.options["llm_models"]
                                ],
                                value=self.default_config["llm_model"],
                            )
                    with gr.Column():
                        with gr.Row():
                            melody_radio = gr.Radio(
                                label="Melody Source",
                                choices=[
                                    (source["name"], source["id"])
                                    for source in self.options["melody_sources"]
                                ],
                                value=self.default_config["melody_source"],
                            )
                        with gr.Row():
                            svs_radio = gr.Radio(
                                label="SVS Model",
                                choices=[
                                    (model["name"], model["id"])
                                    for model in self.options["svs_models"]
                                ],
                                value=self.current_svs_model,
                            )
                        with gr.Row():
                            voice_radio = gr.Radio(
                                label="Singing voice",
                                choices=list(
                                    self.svs_model_map[self.current_svs_model][
                                        "voices"
                                    ].keys()
                                ),
                                value=self.character_info[
                                    self.current_character
                                ].default_voice,
                            )
                character_radio.change(
                    fn=self.update_character,
                    inputs=character_radio,
                    outputs=[character_image, voice_radio],
                )
                asr_radio.change(
                    fn=self.update_asr_model, inputs=asr_radio, outputs=asr_radio
                )
                llm_radio.change(
                    fn=self.update_llm_model, inputs=llm_radio, outputs=llm_radio
                )
                svs_radio.change(
                    fn=self.update_svs_model,
                    inputs=svs_radio,
                    outputs=[svs_radio, voice_radio],
                )
                melody_radio.change(
                    fn=self.update_melody_source,
                    inputs=melody_radio,
                    outputs=melody_radio,
                )
                voice_radio.change(
                    fn=self.update_voice, inputs=voice_radio, outputs=voice_radio
                )
                mic_input.change(
                    fn=self.run_pipeline,
                    inputs=mic_input,
                    outputs=[interaction_log, audio_output],
                )
                metrics_button.click(
                    fn=self.update_metrics,
                    inputs=audio_output,
                    outputs=[metrics_output],
                )

            return demo
        except Exception as e:
            print(f"error: {e}")
            breakpoint()
            return gr.Blocks()

    def update_character(self, character):
        self.current_character = character
        character_voice = self.character_info[self.current_character].default_voice
        self.current_voice = self.svs_model_map[self.current_svs_model]["voices"][
            character_voice
        ]
        return gr.update(value=self.character_info[character].image_path), gr.update(
            value=character_voice
        )

    def update_asr_model(self, asr_model):
        self.pipeline.set_asr_model(asr_model)
        return gr.update(value=asr_model)

    def update_llm_model(self, llm_model):
        self.pipeline.set_llm_model(llm_model)
        return gr.update(value=llm_model)

    def update_svs_model(self, svs_model):
        self.current_svs_model = svs_model
        character_voice = self.character_info[self.current_character].default_voice
        self.current_voice = self.svs_model_map[self.current_svs_model]["voices"][
            character_voice
        ]
        self.pipeline.set_svs_model(
            self.svs_model_map[self.current_svs_model]["model_path"]
        )
        print(
            f"SVS model updated to {self.current_svs_model}. Will set gradio svs_radio to {svs_model} and voice_radio to {character_voice}"
        )
        return (
            gr.update(value=svs_model),
            gr.update(
                choices=list(
                    self.svs_model_map[self.current_svs_model]["voices"].keys()
                ),
                value=character_voice,
            ),
        )

    def update_melody_source(self, melody_source):
        self.current_melody_source = melody_source
        return gr.update(value=self.current_melody_source)

    def update_voice(self, voice):
        self.current_voice = self.svs_model_map[self.current_svs_model]["voices"][voice]
        return gr.update(value=voice)

    def run_pipeline(self, audio_path):
        if not audio_path:
            return gr.update(value=""), gr.update(value="")
        tmp_file = f"audio_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
        results = self.pipeline.run(
            audio_path,
            self.svs_model_map[self.current_svs_model]["lang"],
            self.character_info[self.current_character].prompt,
            self.current_voice,
            output_audio_path=tmp_file,
            max_new_tokens=50,
        )
        formatted_logs = f"ASR: {results['asr_text']}\nLLM: {results['llm_text']}"
        return gr.update(value=formatted_logs), gr.update(
            value=results["output_audio_path"]
        )

    def update_metrics(self, audio_path):
        if not audio_path:
            return gr.update(value="")
        results = self.pipeline.evaluate(audio_path)
        formatted_metrics = "\n".join([f"{k}: {v}" for k, v in results.items()])
        return gr.update(value=formatted_metrics)
