import torch
import torch.nn as nn
from utils.tools import device

class FastSpeech2Loss(nn.Module):
    """ FastSpeech2 Loss """

    def __init__(self, preprocess_config, model_config):
        super(FastSpeech2Loss, self).__init__()
        self.pitch_feature_level = preprocess_config["preprocessing"]["pitch"][
            "feature"
        ]
        self.energy_feature_level = preprocess_config["preprocessing"]["energy"][
            "feature"
        ]
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.cross_entropy_loss = nn.CrossEntropyLoss(ignore_index=-1)

        self.use_variance_predictor = model_config["use_variance_predictor"]
        self.compute_visual_prediction = model_config["visual_prediction"]["compute_visual_prediction"]

        self.use_bert = model_config["bert"]["use_bert"]

    def forward(self, inputs, predictions, no_spectro=False):
        (
            mel_targets,
            _,
            _,
            pitch_targets,
            energy_targets,
            duration_targets,
            phon_align_targets,
            au_targets,
            au_lens_target,
            _,
            emotion_label_vector,
            bert_embs,
        ) = inputs[6:]
        (
            mel_predictions,
            postnet_mel_predictions,
            pitch_predictions,
            energy_predictions,
            log_duration_predictions,
            _,
            src_masks,
            mel_masks,
            _,
            mel_lens,
            _,
            _,
            _,
            phon_align_predictions,
            au_predictions,
            postnet_au_predictions,
            au_masks,
            au_lens_pred,
            src_masks_noSpectro,
            gst_token_attention_scores,
            _,
        ) = predictions
        
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        src_masks_noSpectro = ~src_masks_noSpectro

        log_duration_targets = torch.log(duration_targets.float() + 1)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]

        log_duration_targets.requires_grad = False
        pitch_targets.requires_grad = False
        energy_targets.requires_grad = False
        mel_targets.requires_grad = False
        phon_align_targets.requires_grad = False
        emotion_label_vector.requires_grad = False

        if self.use_bert:
            bert_embs.requires_grad = False

        if no_spectro or (not torch.any(src_masks_noSpectro)):
            mel_loss = torch.Tensor([0]).long().to(device)
            postnet_mel_loss = torch.Tensor([0]).long().to(device)
            pitch_loss = torch.Tensor([0]).long().to(device)
            energy_loss = torch.Tensor([0]).long().to(device)
            duration_loss = torch.Tensor([0]).long().to(device)
            au_loss = torch.Tensor([0]).long().to(device)
            postnet_au_loss = torch.Tensor([0]).long().to(device)
        else:
            if self.use_variance_predictor["pitch"]:
                if self.pitch_feature_level == "phoneme_level":
                    pitch_predictions = pitch_predictions.masked_select(src_masks_noSpectro)
                    pitch_targets = pitch_targets.masked_select(src_masks_noSpectro)
                elif self.pitch_feature_level == "frame_level":
                    pitch_predictions = pitch_predictions.masked_select(mel_masks)
                    pitch_targets = pitch_targets.masked_select(mel_masks)

                
                pitch_loss = self.mse_loss(pitch_predictions, pitch_targets)
            else:
                pitch_loss = torch.Tensor([0]).long().to(device)

            if self.use_variance_predictor["energy"]:
                if self.energy_feature_level == "phoneme_level":
                    # energy_predictions = energy_predictions.masked_select(src_masks)
                    # energy_targets = energy_targets.masked_select(src_masks)
                    energy_predictions = energy_predictions.masked_select(src_masks_noSpectro)
                    energy_targets = energy_targets.masked_select(src_masks_noSpectro)
                if self.energy_feature_level == "frame_level":
                    energy_predictions = energy_predictions.masked_select(mel_masks)
                    energy_targets = energy_targets.masked_select(mel_masks)

                energy_loss = self.mse_loss(energy_predictions, energy_targets)
            else:
                energy_loss = torch.Tensor([0]).long().to(device)

            # log_duration_predictions = log_duration_predictions.masked_select(src_masks)
            # log_duration_targets = log_duration_targets.masked_select(src_masks)
            log_duration_predictions = log_duration_predictions.masked_select(src_masks_noSpectro)
            log_duration_targets = log_duration_targets.masked_select(src_masks_noSpectro)

            duration_loss = self.mse_loss(log_duration_predictions, log_duration_targets)
            # duration_loss = torch.Tensor([0]).long().to(device)

            mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
            postnet_mel_predictions = postnet_mel_predictions.masked_select(
                mel_masks.unsqueeze(-1)
            )
            mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))

            # if mel_targets.numel() > 0:
            mel_loss = self.mae_loss(mel_predictions, mel_targets)
            # mel_loss = torch.Tensor([0]).long().to(device)
            postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
            # postnet_mel_loss = torch.Tensor([0]).long().to(device)
            # else:
            #     mel_loss = torch.Tensor([0]).long().to(device)
            #     postnet_mel_loss = torch.Tensor([0]).long().to(device)

            # if (au_predictions is not None):
            if self.compute_visual_prediction and torch.any(au_masks):
                # try : 
                assert False not in (au_lens_target == au_lens_pred) , ".au in database and after LR are not the same size!"
                # expect AssertionError:
                #     n += 1
                


 # resize au if needed (same size as mel)  
            # if au.shape[0] > sum(duration_au): # mel.shape[0]
            #     au = au[:sum(duration_au) , :]  # au = au[:mel.shape[0], :]
            # elif au.shape[0] < sum(duration_au): # mel.shape[0]
            #     for _ in range(sum(duration_au) - au.shape[0]): # range(mel.shape[0] - au.shape[0])
            #         au = np.concatenate((au, [au[-1, :]]), axis=0)



                au_masks = ~au_masks
                au_targets = au_targets[:, : au_masks.shape[1], :]
                au_masks = au_masks[:, :au_masks.shape[1]]
                au_targets.requires_grad = False


                au_targets = au_targets.masked_select(au_masks.unsqueeze(-1))
                au_predictions = au_predictions.masked_select(au_masks.unsqueeze(-1))

                # if (postnet_au_predictions is not None):
                # postnet_au_predictions = postnet_au_predictions.masked_select(
                #     au_masks.unsqueeze(-1)
                # )

                au_loss = 10*self.mae_loss(au_predictions, au_targets)
                if (postnet_au_predictions is not None):
                    postnet_au_predictions = postnet_au_predictions.masked_select(
                        au_masks.unsqueeze(-1)
                    )

                    postnet_au_loss = 10*self.mae_loss(postnet_au_predictions, au_targets)
                else: 
                    postnet_au_loss = torch.Tensor([0]).long().to(device)
            else:
                au_loss = torch.Tensor([0]).long().to(device)
                postnet_au_loss = torch.Tensor([0]).long().to(device)

        if (phon_align_predictions is not None):
            phon_align_loss = self.cross_entropy_loss(phon_align_predictions, phon_align_targets)
        else:
            phon_align_loss = torch.Tensor([0]).long().to(device)

        if gst_token_attention_scores is not None:
            style_loss = self.cross_entropy_loss(gst_token_attention_scores, emotion_label_vector)
        else:
            style_loss = torch.Tensor([0]).long().to(device)
            
        total_loss = (
            mel_loss + postnet_mel_loss + duration_loss + pitch_loss + energy_loss + phon_align_loss + au_loss + postnet_au_loss + style_loss
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            pitch_loss,
            energy_loss,
            duration_loss,
            phon_align_loss,
            au_loss,
            postnet_au_loss,
            style_loss,
        )