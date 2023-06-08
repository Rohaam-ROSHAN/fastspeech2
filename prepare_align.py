import argparse

import yaml

from preprocessor import  mailabs # RR ,  ljspeech, aishell3, libritts,    3i made a chane, moved te other three modules in comments


def main(config):
    if "LJSpeech" in config["dataset"]:
        ljspeech.prepare_align(config)
    if "AISHELL3" in config["dataset"]:
        aishell3.prepare_align(config)
    if "LibriTTS" in config["dataset"]:
        libritts.prepare_align(config)
    if "M_AILABS" in config["dataset"]:
        mailabs.prepare_align(config)
    if "Blizzard2023" in config["dataset"]:
        mailabs.prepare_align(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
