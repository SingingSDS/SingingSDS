import os
from huggingface_hub import login

if os.getenv("HF_TOKEN"):
    login(token=os.getenv("HF_TOKEN"))

from interface import GradioInterface


def main():
    demo = GradioInterface(
        options_config="config/interface/options.yaml",
        default_config="config/interface/default.yaml",
    ).create_interface()
    demo.launch()


if __name__ == "__main__":
    main()
