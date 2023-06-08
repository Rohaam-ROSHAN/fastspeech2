import os
import json
import copy
import math
from collections import OrderedDict
from regex import B
import copy

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import torch.nn.functional as F
from scipy.io import loadmat

from utils.tools import get_mask_from_lengths, pad

from text import text_to_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReferenceEncoder(nn.Module):
    '''
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    '''

    def __init__(self, preprocess_config, model_config):

        super().__init__()
        
        K = len(model_config["gst"]["conv_filters"])
        filters = [1] + model_config["gst"]["conv_filters"]

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=model_config["gst"]["ref_enc_size"],
                           stride=model_config["gst"]["ref_enc_strides"],
                           padding=model_config["gst"]["ref_enc_pad"]) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=model_config["gst"]["conv_filters"][i]) for i in range(K)])

        out_channels = self.calculate_channels(
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"], 
                model_config["gst"]["ref_enc_size"][0], model_config["gst"]["ref_enc_strides"][0], 
                model_config["gst"]["ref_enc_pad"][0], K)
                
        self.gru = nn.GRU(input_size=model_config["gst"]["conv_filters"][-1] * out_channels,
                          hidden_size=model_config["gst"]["gru_hidden"],
                          batch_first=True)
                          
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.ref_enc_gru_size = model_config["gst"]["gru_hidden"]

    def forward(self, inputs, input_lengths=None):
        out = inputs.view(inputs.size(0), 1, -1, self.n_mel_channels)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        # ------- Memory effectivness (not tested) ----------
        if input_lengths is not None:
            input_lengths = torch.ceil(input_lengths.float() / 2 ** len(self.convs))
            input_lengths = input_lengths.cpu().numpy().astype(int)            
            out = nn.utils.rnn.pack_padded_sequence(
                        out, input_lengths, batch_first=True, enforce_sorted=False)
        # ------- END ----------
                        
        self.gru.flatten_parameters() # initialy commented

        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, L, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L
        
class STL(nn.Module):
    '''
    inputs --- [N, E//2]
    '''

    def __init__(self, model_config):

        super().__init__()
        self.embed = nn.Parameter(torch.FloatTensor(model_config["gst"]["n_style_token"], model_config["gst"]["token_size"] // model_config["gst"]["attn_head"]))
        d_q = model_config["gst"]["gru_hidden"]
        d_k = model_config["gst"]["token_size"] // model_config["gst"]["attn_head"]
        # self.attention = MultiHeadAttention(model_config["gst"]["attn_head"], model_config["gst"]["gru_hidden"], d_k, d_k, dropout=model_config["gst"]["dropout"])
        # self.attention = MultiHeadAttention(query_dim=d_q, key_dim=d_k, num_units=hp.E, num_heads=hp.num_heads)
        self.attention = MultiHeadCrossAttention(
            query_dim=d_q, key_dim=d_k, num_units=model_config["gst"]["token_size"],
            num_heads=model_config["gst"]["attn_head"])
            
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs, target_scores=None):
        if target_scores is None:
            N = inputs.size(0)
            query = inputs.unsqueeze(1)  # [N, 1, E//2]
        else:
            N = target_scores.size(0)
            query = None
            
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, attention_scores = self.attention(query, keys, target_scores)

        return style_embed, attention_scores

class MultiHeadCrossAttention(nn.Module):
    '''
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    '''
    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key, target_scores=None):
        split_size = self.num_units // self.num_heads
        
        if target_scores is None:
            querys = self.W_query(query)  # [N, T_q, num_units]
            querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
            keys = self.W_key(key)  # [N, T_k, num_units]
            keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        
        values = self.W_value(key)
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        if target_scores is None:
            # score = softmax(QK^T / (d_k ** 0.5))
            scores = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
            scores = scores / (self.key_dim ** 0.5)
            scores = F.softmax(scores, dim=3)
        else:
            scores = target_scores.unsqueeze(0).unsqueeze(2)

        # out = score * V
        out = torch.matmul(scores, values)  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]
        
        # scores reshape (when multiple heads, scores concatenate along dim 2, then transpose for cross entropy
        scores = torch.cat(torch.split(scores, 1, dim=0), dim=3).squeeze(0).transpose(1, 2)  # [N, T_k*h=num_units, T_q]
        
        return out, scores
        
class GST(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.encoder = ReferenceEncoder(preprocess_config, model_config)
        self.stl = STL(model_config)

    def forward(self, inputs, input_lengths=None, target_scores=None):
        if target_scores is not None:
            enc_out = None
        else:
            enc_out = self.encoder(inputs, input_lengths=input_lengths)
        style_embed, attention_scores = self.stl(enc_out, target_scores)

        return style_embed, attention_scores
        
class LST(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.embed = nn.Parameter(torch.FloatTensor(model_config["lst"]["n_style_token"], model_config["lst"]["token_size"] // model_config["lst"]["attn_head"]))
        d_q = model_config["transformer"]["encoder_hidden"] + model_config["gst"]["token_size"]
        d_k = model_config["lst"]["token_size"] // model_config["lst"]["attn_head"]
        self.attention = MultiHeadCrossAttention(
            query_dim=d_q, key_dim=d_k, num_units=model_config["lst"]["token_size"],
            num_heads=model_config["lst"]["attn_head"])
            
        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        # local_style_emb, attention_scores = self.stl(enc_out, target_scores)
        
        N = inputs.size(0)
        # query = inputs.unsqueeze(1)  # [N, L, E]
        
        keys = F.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]
        style_embed, attention_scores = self.attention(inputs, keys)

        return style_embed, attention_scores
        
class VarianceAdaptor(nn.Module):
    """Variance Adaptor"""

    def __init__(self, preprocess_config, model_config):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor(model_config)
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor(model_config)
        self.energy_predictor = VariancePredictor(model_config)

        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.pitch_normalization = preprocess_config["preprocessing"]["pitch"][
            "normalization"
        ]
        self.energy_normalization = preprocess_config["preprocessing"]["energy"][
            "normalization"
        ]
        assert self.pitch_feature_level in ["phoneme_level", "frame_level"]
        assert self.energy_feature_level in ["phoneme_level", "frame_level"]

        pitch_quantization = model_config["variance_embedding"]["pitch_quantization"]
        energy_quantization = model_config["variance_embedding"]["energy_quantization"]
        n_bins = model_config["variance_embedding"]["n_bins"]
        assert pitch_quantization in ["linear", "log"]
        assert energy_quantization in ["linear", "log"]
        with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
        ) as f:
            stats = json.load(f)
            pitch_min, pitch_max = stats["pitch"][:2]
            self.pitch_mean, self.pitch_std = stats["pitch"][2:4]
            energy_min, energy_max = stats["energy"][:2]
            self.energy_mean, self.energy_std = stats["energy"][2:4]

        if pitch_quantization == "log":
            self.pitch_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(pitch_min), np.log(pitch_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.pitch_bins = nn.Parameter(
                torch.linspace(pitch_min, pitch_max, n_bins - 1),
                requires_grad=False,
            )
        if energy_quantization == "log":
            self.energy_bins = nn.Parameter(
                torch.exp(
                    torch.linspace(np.log(energy_min), np.log(energy_max), n_bins - 1)
                ),
                requires_grad=False,
            )
        else:
            self.energy_bins = nn.Parameter(
                torch.linspace(energy_min, energy_max, n_bins - 1),
                requires_grad=False,
            )

        self.pitch_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )
        self.energy_embedding = nn.Embedding(
            n_bins, model_config["transformer"]["encoder_hidden"]
        )

        self.use_variance_predictor = model_config["use_variance_predictor"]
        self.use_variance_embeddings = model_config["use_variance_embeddings"]

        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        self.maximum_phoneme_duration = model_config["maximum_phoneme_duration"]

        self.f0_bias_vector = model_config["bias_vector"]["f0"]
        self.f0_bias_coef = model_config["bias_vector"]["coef_f0"]

        self.energy_bias_vector = model_config["bias_vector"]["energy"]
        self.energy_bias_coef = model_config["bias_vector"]["coef_energy"]

        self.spectral_tilt_bias_vector = model_config["bias_vector"]["spectral_tilt"]
        self.spectral_tilt_bias_coef = model_config["bias_vector"]["coef_spectral_tilt"]

        self.duration_bias_vector = model_config["bias_vector"]["duration"]
        self.duration_bias_coef = model_config["bias_vector"]["coef_duration"]

        self.pause_bias_vector = model_config["bias_vector"]["pause"]
        self.liaison_bias_vector = model_config["bias_vector"]["liaison"]
        
        self.embedding_bias = EmbeddingBias(model_config)

        self.factor_interp = ( preprocess_config['preprocessing']['audio']['sampling_rate']/preprocess_config['preprocessing']['stft']['hop_length'] ) / preprocess_config['preprocessing']['au']["sampling_rate"]

    def get_pitch_embedding(self, x, target, mask, control):
        prediction = self.pitch_predictor(x, mask)
        if target is not None:
            embedding = self.pitch_embedding(torch.bucketize(target, self.pitch_bins))
        else:
            # prediction = prediction * control
            if self.pitch_normalization:
                # prediction = prediction * (control + (control-1)*self.pitch_mean/(self.pitch_std*prediction))
                # prediction = prediction + control/self.pitch_std
                prediction = prediction + control/3.5701
            else:
                # prediction = prediction * control
                prediction = prediction + control

            embedding = self.pitch_embedding(
                torch.bucketize(prediction, self.pitch_bins)
            )
            
            batch_size = prediction.size(dim=0)

        return prediction, embedding

    def get_energy_embedding(self, x, target, mask, control):
        prediction = self.energy_predictor(x, mask)
        if target is not None:
            embedding = self.energy_embedding(torch.bucketize(target, self.energy_bins))
        else:
            # prediction = prediction * control
            if self.energy_normalization:
                # prediction = prediction * (control + (control-1)*self.energy_mean/(self.energy_std*prediction))
                # prediction = prediction + control/self.energy_std
                prediction = prediction + control/8.4094
            else:
                # prediction = prediction * control
                prediction = prediction + control
            embedding = self.energy_embedding(
                torch.bucketize(prediction, self.energy_bins)
            )
        return prediction, embedding

    def float2int(self, _duration, x_size):
        _duration_compensated = torch.zeros_like(_duration, dtype=torch.int) # RR, copy.deepcopy(predicted_duration)

        for utt_in_batch in range(x_size[0]):
            residual = 0.0
            for index_phon in range(x_size[1]):
                dur_phon = _duration[utt_in_batch][index_phon]
                dur_phon_rounded = int(torch.round(dur_phon + residual))
                residual += dur_phon - dur_phon_rounded   # RR, actually it is residual = (dur_phon + residual ) - dur_phon_rounded
                _duration_compensated[utt_in_batch][index_phon] = dur_phon_rounded
            
        # Add residual to compensate for round
        duration_rounded = torch.clamp(
            _duration_compensated,
            min=0,
        )
        return duration_rounded


    def forward(
        self,
        x,
        src_mask,
        mel_mask=None,
        # au_mask=None,
        max_mel_len=None, # max_mel_len
        # max_au_len=None,
        pitch_target=None,
        energy_target=None,
        duration_target=None,
        # duration_target_au=None,
        p_control=0.0,
        e_control=0.0,
        d_control=1.0,
        control_bias_array=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        silence_control_bias=False,
        src_mask_noSpectro=None,
    ):

        d_control_bias = control_bias_array[0]
        # addition of a log duration bias
        if d_control_bias != 1.0 and silence_control_bias and False:
            load_bias_vector = loadmat(self.pause_bias_vector) # vector name: vector_attention_duration
            bias_vector = load_bias_vector['vector_attention_duration'].transpose()*(d_control_bias-1)*d_control_bias*2
            bias_vector = bias_vector[:,np.newaxis,:]
            bias_vector = torch.FloatTensor(bias_vector)
            bias_vector = bias_vector.to(device)
            lg_in = x.size()[1]

            x = x + bias_vector.repeat(1,lg_in,1)

        # log_duration_prediction = self.duration_predictor(x, src_mask)
        log_duration_prediction = self.duration_predictor(x, src_mask_noSpectro)
        
        log_duration_prediction_for_au = log_duration_prediction.clone().detach()
        # Here we keep the process go normally, so we dont skip the energy and pitch biasing for now 
        # x_au=x.copy()
        factor_interp = self.factor_interp # 1.4355 # factor_interp = (sampling_rate/hop_length)/au_config["sampling_rate"]
        
        predicted_duration = (torch.exp(log_duration_prediction) - 1) * d_control
        predicted_duration_au = torch.div( (torch.exp(log_duration_prediction_for_au) - 1) * d_control , factor_interp)

        if duration_target is not None:
            duration_target_au = torch.div(  duration_target, factor_interp)  # 1.4355 = ratio between audio SR and visual SR (22050/256) / 60
            duration_target_au = self.float2int(duration_target_au, x.size())
            # duration_target_au = duration_target_au.type(torch.type(int))
            (x_au, au_len) = self.length_regulator(x, duration_target_au)
            duration_rounded_au = duration_target_au
        else:
            
            duration_rounded_au = self.float2int(predicted_duration_au, x.size())
            x_au, au_len = self.length_regulator(x, duration_rounded_au)
        au_mask = get_mask_from_lengths(au_len) # au_mask should be generated in both cases training or inference. since it is new to the model because it is based on duration_*_au which is generated inside this module, not in the model 



        

        if self.pitch_feature_level == "phoneme_level":
            if self.use_variance_predictor["pitch"]:
                # pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                #     x, pitch_target, src_mask, p_control
                # )
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    x, pitch_target, src_mask_noSpectro, p_control
                )
                if self.use_variance_embeddings["pitch"]:
                    x = x + pitch_embedding
            else:
                pitch_prediction = None
                
            # Add Embedding Bias layer 8
            x = self.embedding_bias.layer_control(x, control_bias_array, 8)
            
        output_by_layer = x.unsqueeze(0)

        if self.energy_feature_level == "phoneme_level":
            if self.use_variance_predictor["energy"]:
                # energy_prediction, energy_embedding = self.get_energy_embedding(
                #     x, energy_target, src_mask, e_control
                # )
                energy_prediction, energy_embedding = self.get_energy_embedding(
                    x, energy_target, src_mask_noSpectro, e_control
                )
                if self.use_variance_embeddings["energy"]:
                    x = x + energy_embedding
            else:
                energy_prediction = None
                
            # Add Embedding Bias layer 9
            x = self.embedding_bias.layer_control(x, control_bias_array, 9)

        output_by_layer = torch.cat((output_by_layer, x.unsqueeze(0)), 0)

        if duration_target is not None:
            if self.maximum_phoneme_duration["limit"]: # impose max phon duration
                duration_threshold = self.maximum_phoneme_duration["threshold"]
                duration_target[duration_target>duration_threshold] = duration_threshold
            # print(duration_target)

            # duration_target_au = duration_target / factor_interp  # 1.4355 = ratio between audio SR and visual SR (22050/256) / 60
            # duration_target_au = self.float2int(duration_target_au, x.size())
            # duration_target_au = duration_target_au.type(torch.type(int))
            

            
            (x, mel_len) = self.length_regulator(x, duration_target, max_mel_len) 
            # (x_au, au_len) = self.length_regulator(x, duration_target_au, max_au_len)
            # print(x.size(), x_au.size())
            # x_au, au_len = self.length_regulator(x_au, duration_target_au, max_len)
        

            duration_rounded = duration_target
            # print(duration_target)
            # RR, question: isn't it better to generate again the mel_mask here? to make sure the matches?
        else:
            # duration_rounded = torch.clamp(
            #     (torch.round(torch.exp(log_duration_prediction) - 1) * d_control),
            #     min=0,
            # )

            # predicted_duration = (torch.exp(log_duration_prediction) - 1) * d_control
            # predicted_duration_compensated = predicted_duration

            # for utt_in_batch in range(x.size()[0]):
            #     residual = 0.0
            #     for index_phon in range(x.size()[1]):
            #         dur_phon = predicted_duration[utt_in_batch][index_phon]
            #         dur_phon_rounded = torch.round(dur_phon + residual)
            #         #residual += dur_phon - dur_phon_rounded   # should be changed, instead of +=  shouldn't be just = ? (I guess residual shouldn't be accumulative)
            #         residual = dur_phon_rounded - (dur_phon + residual)
            #         predicted_duration_compensated[utt_in_batch][index_phon] = dur_phon_rounded

            # # Add residual to compensate for round
            # duration_rounded = torch.clamp(
            #     predicted_duration_compensated,
            #     min=0,
            # )
            duration_rounded = self.float2int(predicted_duration, x.size())
            # duration_rounded_au.type()

            x, mel_len = self.length_regulator(x, duration_rounded, max_mel_len)
            # x_au, au_len = self.length_regulator(x, duration_rounded_au, max_au_len)

            mel_mask = get_mask_from_lengths(mel_len)

        if self.pitch_feature_level == "frame_level":
            if self.use_variance_predictor["pitch"]:
                pitch_prediction, pitch_embedding = self.get_pitch_embedding(
                    x, pitch_target, mel_mask, p_control
                )
                if self.use_variance_embeddings["pitch"]:
                    x = x + pitch_embedding
            else:
                pitch_prediction = None

        if self.energy_feature_level == "frame_level":
            if self.use_variance_predictor["energy"]:
                energy_prediction, energy_embedding = self.get_energy_embedding(
                    x, energy_target, mel_mask, e_control
                )
                if self.use_variance_embeddings["energy"]:
                    x = x + energy_embedding
            else:
                energy_prediction = None

        return (
            x,
            x_au,
            pitch_prediction,
            energy_prediction,
            log_duration_prediction,
            duration_rounded,
            mel_len,
            mel_mask,
            output_by_layer,
            self.pitch_embedding,
            self.pitch_bins,
            au_len,# better to be returned
            au_mask, # not necessary for au_mask to be returned
        )



class LengthRegulator(nn.Module):
    """Length Regulator"""

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration_batch, max_mel_len):
        
        output = list()
        mel_len = list()
        for utt, expand_phon_targets in zip(x, duration_batch):
            expanded = self.expand(utt, expand_phon_targets)
            output.append(expanded)
            mel_len.append(expanded.shape[0])

        if max_mel_len is not None:
            output = pad(output, max_mel_len)
        else:
            output = pad(output)
        

        return output, torch.LongTensor(mel_len).to(device)

    def expand(self, utt, expand_phon_targets):
        out = list()
        t = 0

        for i, vec in enumerate(utt):
            expand_size = expand_phon_targets[i].item() # to obtain the value inside the predicted tensor of size 1 
            

            # out.append(vec.expand(max(int(expand_size), 0), -1))
            out.append(vec.expand(max(int(np.round(expand_size)), 0), -1))
            # print(vec.size())
            # print(i, expand_size)
            # print(out[-1].size())
            t += expand_size
            
            


        out = torch.cat(out, 0)
        return out

    def forward(self, x, duration, max_mel_len=None):
        output, mel_len = self.LR(x, duration, max_mel_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """Duration, Pitch and Energy Predictor"""

    def __init__(self, model_config):
        super(VariancePredictor, self).__init__()

        self.input_size = model_config["transformer"]["encoder_hidden"]
        self.filter_size = model_config["variance_predictor"]["filter_size"]
        self.kernel = model_config["variance_predictor"]["kernel_size"]
        self.conv_output_size = model_config["variance_predictor"]["filter_size"]
        self.dropout = model_config["variance_predictor"]["dropout"]

        self.conv_layer = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1d_1",
                        Conv(
                            self.input_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=(self.kernel - 1) // 2,
                        ),
                    ),
                    ("relu_1", nn.ReLU()),
                    ("layer_norm_1", nn.LayerNorm(self.filter_size)),
                    ("dropout_1", nn.Dropout(self.dropout)),
                    (
                        "conv1d_2",
                        Conv(
                            self.filter_size,
                            self.filter_size,
                            kernel_size=self.kernel,
                            padding=1,
                        ),
                    ),
                    ("relu_2", nn.ReLU()),
                    ("layer_norm_2", nn.LayerNorm(self.filter_size)),
                    ("dropout_2", nn.Dropout(self.dropout)),
                ]
            )
        )

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)

        if mask is not None:
            out = out.masked_fill(mask, 0.0)

        return out

class EmbeddingBias(object):
    """
    Bias Module to control acoustic params from embeddings analysis
    """
    def __init__(self, model_config):
        self.bias_vector_name = model_config["bias_vector"]["bias_vector_name"]
        self.layer_by_param = model_config["bias_vector"]["layer_by_param"]
        
    def layer_control(self, embeddings, control_bias_array, layer_index):  
        if control_bias_array==[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]:
            return embeddings
         
        embeddings_size = embeddings.size()
        embeddings_dim = embeddings.dim()
        for index_param, layer_index_by_param in enumerate(self.layer_by_param):
        
            if layer_index_by_param == layer_index:
                load_bias_vector = loadmat(self.bias_vector_name) # vector name: bias_vector_by_layer
                
                bias_vector = load_bias_vector['bias_vector_by_layer'][layer_index_by_param-1][0][:, index_param].transpose()
                bias_size = len(bias_vector)
                
                if index_param == 0:
                    bias_vector = bias_vector*(np.log(control_bias_array[index_param]))
                else:
                    bias_vector = bias_vector*(control_bias_array[index_param])
                    
                #if layer_index==16:
                #    print('yo')
                    
                if embeddings_dim == 2: # frame by frame
                    bias_vector = bias_vector[np.newaxis,:]
                    bias_vector = torch.FloatTensor(bias_vector)
                    bias_vector = bias_vector.to(device)
                    embeddings = embeddings + bias_vector
                else:
                    dim_bias = embeddings_size.index(bias_size)  
                    dim_repeat = 1 if dim_bias==2 else 2
                    lg_repeat = embeddings.size(dim_repeat)
                    
                    if dim_bias == 1:
                        bias_vector = bias_vector[np.newaxis,:,np.newaxis]
                        bias_vector = torch.FloatTensor(bias_vector)
                        bias_vector = bias_vector.to(device)
                        embeddings = embeddings + bias_vector.repeat(1,1,lg_repeat)
                    elif dim_bias == 2:
                        bias_vector = bias_vector[np.newaxis,np.newaxis,:]
                        bias_vector = torch.FloatTensor(bias_vector)
                        bias_vector = bias_vector.to(device)
                        embeddings = embeddings + bias_vector.repeat(1,lg_repeat,1)
                
        return embeddings

class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=0,
        dilation=1,
        bias=True,
        w_init="linear",
    ):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            # padding_mode='replicate',
        )

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x

class LinearNorm(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        # return F.softmax(self.linear_layer(x), dim=2)
        return self.linear_layer(x) # CrossEntropyLoss computes softmax internally
