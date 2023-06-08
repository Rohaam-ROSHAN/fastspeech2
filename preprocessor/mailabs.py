import os

import librosa
import numpy as np
from scipy.io import wavfile
from scipy.interpolate import interp1d
from tqdm import tqdm

from text import _clean_text

import re 

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    csv_dir = config["path"]["csv_path"]
    out_dir = config["path"]["raw_path"]
    preprocessed_dir = config["path"]["preprocessed_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    hop_length = config["preprocessing"]["stft"]["hop_length"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    au_dir = config["path"]["au_path"]
    au_config = config["preprocessing"]["au"]
    os.makedirs((os.path.join(preprocessed_dir, "au")), exist_ok=True)
    csv_file = "AD_all.csv"  # RR , config["path"]["csv_file"] 
    # with open(os.path.join(csv_dir, "NEB_ALN.csv"), encoding="utf-8") as f:
    # with open(os.path.join(csv_dir, "PTT_GB.csv"), encoding="utf-8") as f:
    # with open(os.path.join(csv_dir, "NEB_LVS_ALN.csv"), encoding="utf-8") as f:
    # with open(os.path.join(csv_dir, "all_WAV_ALN_1.csv"), encoding="utf-8") as f:
    with open(os.path.join(csv_dir, csv_file), encoding="utf-8") as f: # RR , 
        for line in tqdm(f):
            parts = line.strip().split("|")
            base_name = parts[0]
            start_utt = parts[1]
            start_utt = int(int(start_utt) * sampling_rate / 1000)
            end_utt = parts[2]
            end_utt = int(int(end_utt) * sampling_rate / 1000)
            text = parts[3]
            text = _clean_text(text, cleaners)
            # align = parts[4]

            if re.match('.*_AD_.*', base_name): # RR , here we should determine the speakers name

                print(base_name)

                #print(len(base_name.split("_")))
                if len(base_name.split("_")) >= 6:
                        speaker = base_name.split("_")[3]
                else:
                        speaker = base_name.split("_")[2]

                # wav_path = os.path.join(in_dir, "_wav_22050", "{}.wav".format(base_name))
                # wav_path = os.path.join(in_dir, "_wav", "{}.wav".format(base_name))
                wav_path = os.path.join(in_dir, "{}.wav".format(base_name))

                number_utt_in_chapter = 1

                if os.path.exists(wav_path):
                    os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                    wav, _ = librosa.load(wav_path, sr=sampling_rate) # I made a change: sr is added, calling by name, not position
                    wav = wav[start_utt:end_utt]
                    wav = wav / max(abs(wav)) * max_wav_value * 0.95

                    while os.path.exists(os.path.join(out_dir, speaker, "{}_{}.wav".format(base_name, number_utt_in_chapter))):
                        number_utt_in_chapter += 1

                    wavfile.write(
                        os.path.join(out_dir, speaker, "{}_{}.wav".format(base_name, number_utt_in_chapter)),
                        sampling_rate,
                        wav.astype(np.int16),
                    )
                    with open(
                        os.path.join(out_dir, speaker, "{}_{}.lab".format(base_name, number_utt_in_chapter)),
                        "w",
                    ) as f1:
                        f1.write(text)

                    # with open(
                    #     os.path.join(out_dir, speaker, "{}_{}.aln".format(base_name, number_utt_in_chapter)),
                    #     "w",
                    # ) as f1:
                    #     f1.write(align)

                    # Write Action Units (AU) as numpy array when exists
                    au_path = os.path.join(au_dir, "{}.AU".format(base_name))
                    if os.path.exists(au_path) and True:    # RR, we changed here, false to true 
                        # while os.path.exists(os.path.join(preprocessed_dir, "au", "{}-au-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter))):
                        #     number_utt_in_chapter += 1

                        start_utt = parts[1]
                        start_utt_au = int(int(start_utt) * au_config["sampling_rate"] / 1000)
                        end_utt = parts[2]
                        end_utt_au = int(int(end_utt) * au_config["sampling_rate"] / 1000)

                        (lg_data_visual, dim_visual, num, den) = tuple(np.fromfile(au_path, count=4, dtype=np.int32))
                        # print(dim_visual)
                        # print(num)
                        # print(den)
                        lg_visual = end_utt_au - start_utt_au
                        # print(lg_visual)
                        visual_params = np.memmap(au_path, offset=16+(start_utt_au * dim_visual * 4), dtype=np.float32, shape=(lg_visual, dim_visual)).transpose()
                        # print(visual_params)
                        # print(visual_params[0,:].size)
                        # print(visual_params[:,0].size)

                    # perform linear interpolation to match mel frame sampling rate
                    # factor_interp = (sampling_rate/hop_length)/au_config["sampling_rate"] # RR, 
                    # # print(factor_interp)
                    # size_interp = round(visual_params[0, :].size*factor_interp) # RR,
                    # visual_params_interp = np.zeros((visual_params[:, 0].size, size_interp)) # RR,
                    # for i in range(0, visual_params[:,0].size): 
                        # visual_params_interp[i] = np.interp(np.linspace(0, 1, size_interp), np.linspace(0, 1, visual_params[0, :].size), visual_params[i, :])
                    # # print(visual_params_interp[0,:].size)


                        au_filename = "{}-au-{}_{}.npy".format(speaker, base_name, number_utt_in_chapter)
                        np.save(
                            os.path.join(preprocessed_dir, "au", au_filename),
                            visual_params.transpose(),
                        )
