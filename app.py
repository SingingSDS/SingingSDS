from interface import GradioInterface


def main():
    demo = GradioInterface(
        options_config="config/options.yaml", default_config="config/default.yaml"
    ).create_interface()
    demo.launch()


if __name__ == "__main__":
    main()
