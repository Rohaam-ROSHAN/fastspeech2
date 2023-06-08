import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from transformer import Encoder, Decoder, PostNet, DecoderVisual
from .modules import VarianceAdaptor, LinearNorm, EmbeddingBias, GST, LST
from utils.tools import get_mask_from_lengths, get_mask_from_lengths_noSpectro
from scipy.io import loadmat

from text.symbols import out_symbols
from text import _find_pattern_indexes_in_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2, self).__init__()
        self.model_config = model_config

        self.encoder = Encoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.postnet = PostNet(n_mel_channels=preprocess_config["preprocessing"]["mel"]["n_mel_channels"])
        self.compute_phon_prediction = model_config["compute_phon_prediction"]
        self.compute_visual_prediction = model_config["visual_prediction"]["compute_visual_prediction"]
        self.visual_postnet = model_config["visual_prediction"]["visual_postnet"]
        self.separate_visual_decoder = model_config["visual_prediction"]["separate_visual_decoder"]

        # Phonetic prediction from input
        if self.compute_phon_prediction:   # RR, it is true now
            self.dim_out_symbols = len(out_symbols)
            self.phonetize = LinearNorm(model_config["transformer"]["encoder_hidden"], self.dim_out_symbols)

        # Action Units prediction
        if self.compute_visual_prediction: # RR, it is true now
            if self.separate_visual_decoder: # RR, it is true now
                self.decoder_visual = DecoderVisual(model_config)
                self.au_linear = nn.Linear(
                    model_config["visual_decoder"]["decoder_hidden"],
                    preprocess_config["preprocessing"]["au"]["n_units"],
                )
            else:
                self.au_linear = nn.Linear(
                    model_config["transformer"]["decoder_hidden"],
                    preprocess_config["preprocessing"]["au"]["n_units"],
                )

            if self.visual_postnet:
                self.postnet_visual = PostNet(n_mel_channels=preprocess_config["preprocessing"]["au"]["n_units"])

        self.speaker_emb = None
        if model_config["multi_speaker"]:
            with open(
                os.path.join(
                    preprocess_config["path"]["preprocessed_path"], "speakers.json"
                ),
                "r",
            ) as f:
                n_speaker = len(json.load(f))
            self.speaker_emb = nn.Embedding(
                n_speaker,
                model_config["transformer"]["encoder_hidden"],
            )

        self.pause_bias_vector = model_config["bias_vector"]["pause"]
        self.liaison_bias_vector = model_config["bias_vector"]["liaison"]
        
        self.embedding_bias = EmbeddingBias(model_config)
        
        self.use_gst = model_config["gst"]["use_gst"]
        if self.use_gst:
            self.gst = GST(preprocess_config, model_config)
            
        self.use_lst = model_config["lst"]["use_lst"]
        if self.use_lst:
            self.lst = LST(model_config)

        self.use_bert = model_config["bert"]["use_bert"]
        if self.use_bert:
            self.bert_adaptation_layer = nn.Linear(
                model_config["bert"]["bert_size_org"],
                model_config["bert"]["bert_size_reduced"],
            )

    def freeze_encoder(self):
        print('Freeze_encoder')
        for name, child in self.encoder.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # # Freeze char embeddings
        # for param in self.embedding.parameters():
        #     param.requires_grad = False

    def freeze_decoder(self):
        print('Freeze_decoder')
        for name, child in self.decoder.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # Freeze Mel Linear
        for param in self.mel_linear.parameters():
            param.requires_grad = False
    
    def freeze_decoder_visual(self):
        print('Freeze_decoder_visual')
        for name, child in self.decoder_visual.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # Freeze Mel Linear
        for param in self.au_linear.parameters():
            param.requires_grad = False

    def freeze_postnet(self):
        print('Freeze_postnet')
        for name, child in self.postnet.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        
    def freeze_postnet_visual(self):
        print('Freeze_postnet_visual')
        for name, child in self.postnet_visual.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
    
    def freeze_speaker_emb(self):
        print('Freeze_Speaker_Embeddings')
        for param in self.speaker_emb.parameters():
            param.requires_grad = False

    def freeze_phon_prediction(self):
        print('Freeze_Phon_Prediction')
        for name, child in self.phonetize.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        
    def freeze_variance_prediction(self):
        print('Freeze_Variance_Prediction')
        for name, child in self.variance_adaptor.named_children():
            # print('{} {}'.format(name,child))
            for param in child.parameters():
                param.requires_grad = False
        # print('Freeze Prosodic Embeddings')
        # for param in self.embedding.parameters():
        #     param.requires_grad = False

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels=None,
        mel_lens=None,
        max_mel_len=None,
        p_targets=None,
        e_targets=None,
        d_targets=None,
        phon_align_targets=None,
        au_targets=None,
        au_lens=None,
        max_au_len=None,
        emotion_vector=None,
        bert_embs=None,
        inference_gst_token_vector=None,
        p_control=0.0,
        e_control=0.0,
        d_control=1.0,
        control_bias_array=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        pause_control_bias=0.0,
        liaison_control_bias=0.0,
        silence_control_bias=False,
        no_spectro=False,
    ):
        #print(bert_embs)
        #p_targets=None
        #e_targets=None
        #d_targets=None
        
        src_masks = get_mask_from_lengths(src_lens, max_src_len)

        src_masks_noSpectro = get_mask_from_lengths_noSpectro(src_lens, mel_lens, max_src_len)
        
        mel_masks = (
            get_mask_from_lengths(mel_lens, max_mel_len)
            if mel_lens is not None
            else None
        )
        au_masks = (
            get_mask_from_lengths(au_lens, max_au_len) # RR, Why max_mel_len and not max_au_len ? I changed it!
            if au_lens is not None
            else None
        )

        if self.use_bert:
            reduced_bert = self.bert_adaptation_layer(bert_embs)
        else:
            reduced_bert = None

        output, enc_output_by_layer = self.encoder(texts, src_masks, control_bias_array=control_bias_array, reduced_bert=reduced_bert)
        
        # GST
        if self.use_gst:
            style_embedding, gst_token_attention_scores = self.gst(mels, mel_lens, inference_gst_token_vector)
            #print(style_embedding.shape)
            #print(gst_token_attention_scores.shape)
            #print(output.shape)
            
            style_emb_output = style_embedding.expand(
                -1, max_src_len, -1
            )

            if not self.use_lst:
                output = output + style_emb_output
        else:
            gst_token_attention_scores = None

        if self.speaker_emb is not None:
            output = output + self.speaker_emb(speakers).unsqueeze(1).expand(
                -1, max_src_len, -1
            )
            enc_output_by_layer = torch.cat((enc_output_by_layer, output.unsqueeze(0)), 0)
        else:
            enc_output_by_layer = torch.cat((enc_output_by_layer, enc_output_by_layer[-1, :, :, :].unsqueeze(0)), 0)
            
        # Add Embedding Bias layer 7
        output = self.embedding_bias.layer_control(output, control_bias_array, 7)
        
        # LST
        if self.use_lst:
            local_style_embedding, lst_token_attention_scores = self.lst(torch.cat((output, style_emb_output), 2))
            #print(local_style_embedding.shape)
            #print(lst_token_attention_scores.shape)
            #print(output.shape)
            
            output = output + local_style_embedding
        else:
            lst_token_attention_scores = None
            
        # Add Pauses between words
        if pause_control_bias != 0.0:
            # Load Bias vector 
            load_bias_vector = loadmat(self.pause_bias_vector) # vector name: vector_attention_duration
            bias_vector = load_bias_vector['vector_attention_duration'].transpose()*pause_control_bias
            zero_bias_vector = np.zeros([output.size(0), output.size(1), output.size(2)])

            list_patterns = np.array([
                (' ', 0),
                (',', 0),
                ('.', 0),
                ('?', 0),
                ('!', 0),
                (':', 0),
                (';', 0),
                ('§', 0),
                ('~', 0),
                ('[', 0),
                (']', 0),
                ('(', 0),
                (')', 0),
                ('-', 0),
                ('"', 0),
                ('¬', 0),
                ('«', 0),
                ('»', 0),
            ])
            [index_utt_in_batch, index_target_char_in_utt] = _find_pattern_indexes_in_batch(list_patterns, texts)
            zero_bias_vector[index_utt_in_batch, index_target_char_in_utt, :] = bias_vector
            
            zero_bias_vector = torch.FloatTensor(zero_bias_vector)
            zero_bias_vector = zero_bias_vector.to(device)
            output = output + zero_bias_vector

        # Add Liaisons
        if liaison_control_bias != 0.0:
            # Load Bias vector 
            load_bias_vector = loadmat(self.liaison_bias_vector) # vector name: vector_liaison
            bias_vector = load_bias_vector['vector_liaison'].transpose()*(-liaison_control_bias)
            zero_bias_vector = np.zeros([output.size(0), output.size(1), output.size(2)])

            list_patterns = np.array([
                ('er a', 1),
                ('er à', 1),
                ('er e', 1),
                ('er i', 1),
                ('er o', 1),
                ('er u', 1),
                ('er y', 1),
                ('t a', 0),
                ('t e', 0),
                ('t i', 0),
                ('t o', 0),
                ('t u', 0),
                ('t y', 0),
                ('n a', 0),
                ('n â', 0),
                ('n e', 0),
                ('n i', 0),
                ('n o', 0),
                ('n u', 0),
                ('n y', 0),
                ('es a', 1),
                ('es e', 1),
                ('es i', 1),
                ('es o', 1),
                ('es u', 1),
                ('es y', 1),
            ])
            [index_utt_in_batch, index_target_char_in_utt] = _find_pattern_indexes_in_batch(list_patterns, texts)
            zero_bias_vector[index_utt_in_batch, index_target_char_in_utt, :] = bias_vector
            
            zero_bias_vector = torch.FloatTensor(zero_bias_vector)
            zero_bias_vector = zero_bias_vector.to(device)
            output = output + zero_bias_vector
            
        if self.compute_phon_prediction:
            phon_outputs = self.phonetize(output).transpose(1,2)
        else:
            phon_outputs = None

        # If no_spectro, only train the encoder with the phonetic prediction, do not compute spectro
        if no_spectro:
            return (
                    None,
                    None,
                    None,
                    None,
                    None,
                    None,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                    enc_output_by_layer,
                    None, # RR, it seems that one None is redundant
                    None,
                    None,
                    phon_outputs,
                    None,
                    None,
                    au_masks,
                    au_lens,
                )

        (
            output,
            output_au,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            # d_rounded_au,
            mel_lens,
            mel_masks,
            # au_masks,
            output_by_layer_variance_adaptor,
            pitch_embeddings,
            pitch_bins,
            au_lens,
            au_masks,
        ) = self.variance_adaptor(
            output,
            src_masks,
            mel_masks, # get_masks_from_length( . . . )
            # au_masks,
            max_mel_len, # comes from input of forward, efore from dataset   
            # max_au_len,
            p_targets,
            e_targets,
            d_targets,
            # d_targets_au,
            p_control,
            e_control,
            d_control,
            control_bias_array,
            silence_control_bias,
            src_masks_noSpectro,
        )

        enc_output_by_layer = torch.cat((enc_output_by_layer, output_by_layer_variance_adaptor), 0)
        
        # output_au = output # For visual prediction


        if mel_masks.nelement():
            output, mel_masks, dec_output_by_layer = self.decoder(output, mel_masks, control_bias_array=control_bias_array)
        else: 
            dec_output_by_layer = None

        # Action Units prediction
        if self.compute_visual_prediction:
            if self.separate_visual_decoder:
                if au_masks.nelement(): 
                    output_au, au_masks, visual_dec_output_by_layer = self.decoder_visual(output_au, au_masks)
                    output_au = self.au_linear(output_au)
            else:
                output_au = self.au_linear(output)
                
            au_output_by_layer = output_au.unsqueeze(0)

            if self.visual_postnet:
                if au_masks.nelement(): 
                    postnet_output_au, postnet_output_by_layer_au = self.postnet_visual(output_au)
                    postnet_output_au = postnet_output_au + output_au
                    au_output_by_layer = torch.cat((au_output_by_layer, postnet_output_au.unsqueeze(0)), 0)
                else:
                    postnet_output_au = output_au
                    au_output_by_layer = torch.cat((au_output_by_layer, au_output_by_layer[-1, :, :, :].unsqueeze(0)), 0)
            else:
                postnet_output_au = None
                postnet_output_by_layer_au = None
                au_output_by_layer = torch.cat((au_output_by_layer, au_output_by_layer[-1, :, :, :].unsqueeze(0)), 0)
        else:
            output_au = None
            postnet_output_au = None
            visual_dec_output_by_layer = None
            postnet_output_by_layer_au = None
            au_output_by_layer = None
            au_masks = None

        output = self.mel_linear(output)
        mel_output_by_layer = output.unsqueeze(0)
        
        if mel_masks.nelement(): 
            postnet_output, postnet_output_by_layer = self.postnet(output)
            postnet_output = postnet_output + output
        else:
            postnet_output = output
            postnet_output_by_layer = None
        
        mel_output_by_layer = torch.cat((mel_output_by_layer, postnet_output.unsqueeze(0)), 0)

        return (
            output,
            postnet_output,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            src_masks,
            mel_masks,
            src_lens,
            mel_lens,
            [enc_output_by_layer, dec_output_by_layer, postnet_output_by_layer, mel_output_by_layer, visual_dec_output_by_layer, postnet_output_by_layer_au, au_output_by_layer],
            pitch_embeddings,
            pitch_bins,
            phon_outputs,
            output_au,
            postnet_output_au,
            au_masks,
            au_lens,
            src_masks_noSpectro,
            gst_token_attention_scores,
            lst_token_attention_scores,
        )
