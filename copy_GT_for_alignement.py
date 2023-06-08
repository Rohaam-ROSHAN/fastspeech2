import argparse
import yaml
import os
from tqdm import tqdm

import numpy as np
from scipy.io import savemat

from preprocessor import ljspeech, aishell3, libritts, mailabs


def main(config):
    input_file = "train_phon.txt"
    output_path = "_NEB_Fastspeech2_120000_phonPred_predictor_embeddings_train_GT"

    raw_data = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    output_syn_path = config["path"]["output_syn_path"]

    with open(os.path.join(preprocessed_dir, input_file), encoding="utf-8") as f:
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            speaker = parts[1]
            text_utt = parts[2]
            raw_utt = parts[3]
            phon_align = parts[4]

            print(base_name)

            # Copy audio file
            input_audio_file = os.path.join(raw_data, speaker, "{}.wav".format(base_name))
            output_audio_file = os.path.join(output_syn_path, output_path, "{}.wav".format(base_name))
            os.popen('cp {} {}'.format(input_audio_file, output_audio_file))

            # Copy duration mat
            input_duration_file = os.path.join(preprocessed_dir, "duration", "{}-duration-{}.npy".format(speaker, base_name))
            duration_target = np.load(input_duration_file)

            # save duration_target in .mat format
            mdic = {"duration_mat": duration_target}
            nm_duration = '{}/{}/{}_duration_target.mat'.format(output_syn_path, output_path, base_name)
            savemat(nm_duration, mdic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to preprocess.yaml")
    args = parser.parse_args()

    config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    main(config)
