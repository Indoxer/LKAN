from pathlib import Path

from lkan.runner import Runner


def main():
    path = Path(__file__).parent
    config_path = f"{path}/lkan/configs/kan_linear/mnist.yaml"

    runner = Runner()
    cfg = runner.load_config(config_path)
    runner.run(cfg)


if __name__ == "__main__":
    main()
