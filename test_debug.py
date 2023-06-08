import torch
import sys

a=torch.zeros(5, 6)
args = sys.argv

print('salut')
for arg in args :
    print(arg)

   args =  ["-p" "config/M_AILABS/preprocess.yaml" "-m" "config/M_AILABS/model_emotion_AD.yaml" "-t" "config/M_AILABS/train_emotion_AD_14tokens.yaml"]