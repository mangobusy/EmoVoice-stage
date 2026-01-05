from pyexpat import model
import torch
import torch.nn as nn
import os
import logging
import torch.nn.functional as F
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src')))
from slam_llm.models.slam_model import (
    slam_model,
    setup_tokenizer,
    setup_llm,
)
from slam_llm.utils.train_utils import print_model_size
from typing import List, Optional
from slam_llm.utils.metric import compute_accuracy
from tqdm import tqdm
from utils.codec_utils import setup_codec
from utils.codec_utils import layershift as layer_shift, simple_shift
from utils.projector_utils import setup_group_decode_adapter
from slam_llm.utils.config_utils import generate_peft_config
from peft import get_peft_model

logger = logging.getLogger(__name__)

def model_factory(train_config, model_config, **kwargs):
    # return necessary components for training
    tokenizer = setup_tokenizer(train_config, model_config, **kwargs)
    
    encoder = None
    encoder_projector = None

    llm = setup_llm(train_config, model_config, **kwargs)

    codec_decoder = None
    if model_config.codec_decode:
        codec_decoder = setup_codec(train_config, model_config, **kwargs)

    group_decode_adapter = None
    if model_config.group_decode:
        group_decode_adapter = setup_group_decode_adapter(model_config, train_config, **kwargs)
        if train_config.freeze_group_decode_adapter:
            for name, param in group_decode_adapter.named_parameters():
                param.requires_grad = False
            group_decode_adapter.eval()

    model = slam_model_tts(
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    )

    ckpt_path = kwargs.get("ckpt_path", None)
    if ckpt_path is not None:
        logger.info("loading other parts from: {}\n".format(ckpt_path))
        ckpt_dict = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)

    if train_config.use_peft:
        logger.info("setup peft for llm")
        peft_config = generate_peft_config(train_config)
        model.llm = get_peft_model(model.llm, peft_config)
        if int(os.environ.get("RANK", "0")) == 0:
            model.llm.print_trainable_parameters()

    if kwargs.get("peft_ckpt", None):
        logger.info("loading peft-stage ckpt from: {}\n".format(kwargs.get("peft_ckpt")))
        ckpt_dict = torch.load(kwargs.get("peft_ckpt"), map_location="cpu")
        model.load_state_dict(ckpt_dict, strict=False)
    
    # Load stage1 checkpoint for stage2 training
    if hasattr(train_config, 'training_stage') and train_config.training_stage == 2:
        stage1_ckpt = kwargs.get("stage1_ckpt", None)
        if stage1_ckpt is not None:
            logger.info("Loading stage1 checkpoint from: {}\n".format(stage1_ckpt))
            ckpt_dict = torch.load(stage1_ckpt, map_location="cpu")
            model.load_state_dict(ckpt_dict, strict=False)
        
        # Optionally freeze emotion predictor in stage 2
        if hasattr(train_config, 'freeze_emotion_predictor') and train_config.freeze_emotion_predictor:
            logger.info("Freezing emotion predictor for stage 2 training")
            for param in model.emotion_predictor.parameters():
                param.requires_grad = False
            model.emotion_predictor.eval()

    # ==================== 修复代码 START ====================
    # 获取当前训练阶段，默认为 2
    current_stage = getattr(train_config, 'training_stage', 2)
    
    # 如果是 Stage 2，必须冻结 emotion_predictor，否则 DDP 会报错 unused parameters
    if str(current_stage) == "2":
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info("Stage 2 Training: Freezing emotion_predictor parameters to avoid DDP error.")
            
        for param in model.emotion_predictor.parameters():
            param.requires_grad = False
        
        # 确保它处于 eval 模式 (关闭 Dropout 等)
        model.emotion_predictor.eval()
    # ==================== 修复代码 END ======================


    print_model_size(
        model,
        train_config,
        (
            int(os.environ["RANK"])
            if train_config.enable_fsdp or train_config.enable_ddp
            else 0
        ),
    )
    return model, tokenizer


class EmotionPredictor(nn.Module):
    """
    Emotion prediction head for predicting continuous arousal (A) and valence (V) values.
    """
    def __init__(self, hidden_size, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size // 2)
        self.emotion_head = nn.Linear(hidden_size // 2, 2)  # Output: [arousal, valence]
        self.activation = nn.ReLU()
        
    def forward(self, hidden_states):
        # hidden_states: [batch_size, seq_len, hidden_size]
        # Use mean pooling over sequence dimension
        pooled_output = torch.mean(hidden_states, dim=1)  # [batch_size, hidden_size]
        
        x = self.dense1(pooled_output)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.activation(x)
        x = self.dropout(x)
        emotion_logits = self.emotion_head(x)  # [batch_size, 2]
        
        # Apply sigmoid to constrain output to [0, 1]
        emotion_values = torch.sigmoid(emotion_logits)
        
        return emotion_values


class slam_model_tts(slam_model):
    def __init__(
        self,
        encoder,
        llm,
        encoder_projector,
        tokenizer,
        codec_decoder,
        group_decode_adapter,
        train_config,
        model_config,
        **kwargs,
    ):
        super().__init__(
            encoder,
            llm,
            encoder_projector,
            tokenizer,
            train_config,
            model_config,
            **kwargs,
        )

        # resize llm embedding layer
        self.original_vocabsize = self.llm.lm_head.weight.size(0)
        if self.model_config.vocab_config.total_vocabsize != self.original_vocabsize:
            self.llm.resize_token_embeddings(self.model_config.vocab_config.total_vocabsize)
            if int(os.environ.get("RANK", "0")) == 0:
                logger.info("Resize llm embedding layer's vocab size to {}\n".format(self.model_config.vocab_config.total_vocabsize))

        self.codec_decoder = codec_decoder
        self.code_layer = self.model_config.vocab_config.code_layer
        self.group_decode_adapter = group_decode_adapter
        
        # Initialize emotion predictor for stage 1 training
        if hasattr(self.llm.config, 'hidden_size'):
            print("Using hidden_size from llm.config")
            hidden_size = self.llm.config.hidden_size
        elif hasattr(self.llm.config, 'n_embd'):
            print("Using n_embd from llm.config")
            hidden_size = self.llm.config.n_embd
        else:
            print("Using default hidden_size = 869")
            hidden_size = 869  # default value
            
        self.emotion_predictor = EmotionPredictor(
            hidden_size=hidden_size,
            dropout=getattr(model_config, 'emotion_dropout', 0.1)
        )
        
        # Get training stage (default to 2 for backward compatibility)
        self.training_stage = getattr(train_config, 'training_stage', 2)
        
        if int(os.environ.get("RANK", "0")) == 0:
            logger.info(f"Initialized model for training stage: {self.training_stage}\n")

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        modality_mask = kwargs.get("modality_mask", None)
        encoder_outs = None
        
        # Get emotion labels if provided (for stage 1)
        emotion_labels = kwargs.get("emotion_labels", None)  # [batch_size, 2] (A, V)

        if input_ids is not None:
            assert input_ids.min() >= 0, f"input_ids min = {input_ids.min()}"
            assert input_ids.max() < self.model_config.vocab_config.total_vocabsize, \
                f"input_ids max = {input_ids.max()}, vocab = {self.model_config.vocab_config.total_vocabsize}"



        if input_ids is not None:
            input_ids[input_ids == -1] = 0  # [btz, code_layer + 1, seq_length]
            if hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
                # print("==================llm.model.embed_tokens", inputs_embeds)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)
        # print("==================before",inputs_embeds)
        if modality_mask is not None and encoder_outs is not None:
            if self.train_config.modeling_paradigm == "parallel":
                modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
                modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    for j in range(self.code_layer):
                        start_idx = modality_mask_start_indices[i, j].item()
                        length = modality_lengths[i][j]
                        encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
                
                inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
                inputs_embeds = torch.mean(inputs_embeds, dim=1)
            
                # print("==================parallel",inputs_embeds)


            elif self.train_config.modeling_paradigm == "interleaved":
                inputs_embeds = inputs_embeds.squeeze(1)
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
                modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    encoder_outs_pad[
                        i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                    ] = encoder_outs[i][:modality_lengths[i]]
                
                inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
                # print("==================interleaved",inputs_embeds)

            else:
                raise NotImplementedError
            
        # print("==================after",inputs_embeds)
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layer

        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask

        # ==================== STAGE 1: EMOTION PREDICTION ====================
        if self.training_stage == 1:
            # Forward through LLM to get hidden states
            # We only use text input for emotion prediction
            text_labels_stage1 = labels[:, self.code_layer] if labels is not None and labels.dim() > 2 else labels
            
            model_outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=text_labels_stage1,
                output_hidden_states=True
            )
            
            # Get last hidden state for emotion prediction
            hidden_states = model_outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
            
            # Predict emotion values
            emotion_pred = self.emotion_predictor(hidden_states)  # [batch_size, 2]
            
            # Compute emotion prediction loss (MSE)
            emotion_loss = torch.tensor(0.0, device=inputs_embeds.device)
            emotion_mae = torch.tensor(0.0, device=inputs_embeds.device)
            
            if emotion_labels is not None:
                # emotion_labels: [batch_size, 2] with values in [0, 1]
                emotion_loss = F.mse_loss(emotion_pred, emotion_labels)
                
                # Also compute MAE for monitoring
                with torch.no_grad():
                    emotion_mae = F.l1_loss(emotion_pred, emotion_labels)
            
            # Optionally combine with language modeling loss
            use_lm_loss_stage1 = getattr(self.train_config, 'use_lm_loss_stage1', False)
            lm_loss_weight = getattr(self.train_config, 'lm_loss_weight_stage1', 0.1)
            
            if use_lm_loss_stage1 and model_outputs.loss is not None:
                total_loss = emotion_loss + lm_loss_weight * model_outputs.loss
            else:
                total_loss = emotion_loss
            
            model_outputs.loss = total_loss
            loss_recorder = [emotion_loss.item(), emotion_mae.item()]
            # Return emotion prediction results
            return model_outputs, emotion_pred, emotion_mae.item(), loss_recorder
        
        # ==================== STAGE 2: SPEECH GENERATION (ORIGINAL) ====================
        else:
            
            
            kwargs["emotion_labels"] = None
            if self.train_config.modeling_paradigm == "serial":
                temp_labels = labels[:,self.code_layer - 1] if labels is not None else None
                model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=temp_labels)

                eot = self.model_config.vocab_config.eot
                text_pad_token = self.model_config.vocab_config.pad_t
                audio_pad_token = self.model_config.vocab_config.pad_a
                text_labels = torch.full_like(labels, -100)
                audio_labels = torch.full_like(labels, -100)
                batch_size, seq_size, length = labels.shape
                for i in range(batch_size):
                    eot_position = (labels[i, 0] == eot).nonzero(as_tuple=True)[0]
                    eot_pos = eot_position.item()  
                    text_labels[i, :, :eot_pos+1] = labels[i, :, :eot_pos+1]
                    text_labels[i, :, eot_pos+1:] = text_pad_token

                    audio_labels[i, :, eot_pos+1:] = labels[i, :, eot_pos+1:]
                    ignore_pos = torch.where(labels[i, 0]!= -100)[0][0].item()
                    audio_labels[i, :, ignore_pos:eot_pos+1] = audio_pad_token
                text_labels = text_labels[:, 0, :]
            else:
                text_labels = labels[:,self.code_layer] if labels is not None else None
                audio_labels = labels[:, :self.code_layer] if labels is not None else None
                model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=text_labels)

            if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
                x_ori = model_outputs.logits
                text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
                audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
                xt = x_ori[..., :text_vocab_size]
                xa = []

                if self.group_decode_adapter is not None:
                    x_audio_ori = x_ori[..., text_vocab_size:]
                    x_audio = self.group_decode_adapter(x_audio_ori)
                    for i in range(self.code_layer):
                        xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
                else:
                    for i in range(self.code_layer):
                        xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])

                loss_recorder = []
                total_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels)
                
                # Optionally add emotion loss in stage 2
                use_emotion_loss_stage2 = getattr(self.train_config, 'use_emotion_loss_stage2', False)
                if use_emotion_loss_stage2 and emotion_labels is not None:
                    hidden_states = model_outputs.hidden_states[-1] if hasattr(model_outputs, 'hidden_states') and model_outputs.hidden_states else None
                    if hidden_states is not None:
                        emotion_pred = self.emotion_predictor(hidden_states)
                        emotion_loss = F.mse_loss(emotion_pred, emotion_labels)
                        emotion_loss_weight = getattr(self.train_config, 'emotion_loss_weight_stage2', 0.1)
                        total_loss = total_loss + emotion_loss_weight * emotion_loss
                
                model_outputs.loss = total_loss
            elif self.train_config.modeling_paradigm == "interleaved":
                x_ori = model_outputs.logits
            else:
                raise NotImplementedError

            text_acc = -1
            audio_acc = [-1 for _ in range(self.code_layer)] if self.code_layer > 0 else -1
            if self.metric:
                with torch.no_grad():
                    if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
                        preds = torch.argmax(xt, -1)
                        text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                        preds_audio = [torch.argmax(xa[i], -1) for i in range(self.code_layer)]
                        audio_acc = [compute_accuracy(preds_audio[i].detach()[:, :-1], audio_labels[:, i, 1:], ignore_label=-100) for i in range(self.code_layer)]
                    elif self.train_config.modeling_paradigm == "interleaved":
                        preds_start_idx = (text_labels != -100).float().argmax(dim=1)
                        preds = torch.argmax(x_ori, -1)
                        text_preds, text_labels, audio_preds, audio_labels = self.extract_interleaved_tokens(preds, text_labels, preds_start_idx) 

                        text_pad_token = self.model_config.vocab_config.pad_t
                        text_labels[text_labels == text_pad_token] = -100
                        
                        text_acc = compute_accuracy(text_preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)

                        audio_pad_token = self.model_config.vocab_config.pad_a
                        audio_labels[audio_labels == audio_pad_token] = -100
                        
                        audio_acc = [compute_accuracy(audio_preds.detach()[:, :-1], audio_labels.detach()[:, 1:], ignore_label=-100)]

                        loss_recorder = None
                    else:
                        raise NotImplementedError

            return model_outputs, text_acc, audio_acc, loss_recorder

    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss for text and audio layers.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer+1)] 
        
        if text_labels is not None:
            text_loss = F.cross_entropy(xt[:, :-1, :].reshape(-1, text_vocab_size), text_labels[:, 1:].reshape(-1), ignore_index=-100)
            layer_loss[self.code_layer] = text_loss
        else:
            text_loss = 0

        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels[:,i] is not None:
                single_audio_loss = F.cross_entropy(xa[i][:, :-1, :].reshape(-1, audio_vocab_size), audio_labels[:, i, 1:].reshape(-1), ignore_index=-100)
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        total_loss = (text_loss + total_audio_loss) / (self.code_layer+1)
        return total_loss, layer_loss

    @torch.no_grad()
    def predict_emotion(self, input_ids, attention_mask=None):
        """
        Predict emotion values (arousal, valence) from text input.
        For inference/evaluation purposes.
        """
        if input_ids is not None:
            input_ids_copy = input_ids.clone()
            input_ids_copy[input_ids_copy == -1] = 0
            
            if hasattr(self.llm.model, "embed_tokens"):
                inputs_embeds = self.llm.model.embed_tokens(input_ids_copy)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids_copy)
            else:
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids_copy)
            
            inputs_embeds = torch.mean(inputs_embeds, dim=1)
        
        # Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states and predict emotions
        hidden_states = outputs.hidden_states[-1]
        emotion_pred = self.emotion_predictor(hidden_states)
        
        return emotion_pred  # [batch_size, 2] - (arousal, valence)

    # Keep all existing generation methods unchanged
    @torch.no_grad()
    def generate(self, *args, **kwargs):
        # Original generate method remains unchanged
        return super().generate(*args, **kwargs) if hasattr(super(), 'generate') else self._original_generate(*args, **kwargs)
    
    def _original_generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer + 1)]
        current_input_text = None
        current_audio_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize

        num_latency_tokens = kwargs.get("num_latency_tokens", 0)
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        upsampling_factor = kwargs.get("upsampling_factor", 1)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift

        pad_t = self.model_config.vocab_config.pad_t
        pad_a = self.model_config.vocab_config.pad_a
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False

        if self.train_config.modeling_paradigm == "interleaved":
            model_outputs = self.llm.generate(
                inputs_embeds=inputs_embeds,
                max_new_tokens=max_new_tokens,
                num_beams=kwargs.get("num_beams", 4),
                do_sample=kwargs.get("do_sample", False),
                min_length=kwargs.get("min_length", 1),
                top_p=kwargs.get("top_p", 1.0),
                repetition_penalty=text_repetition_penalty,
                length_penalty=kwargs.get("length_penalty", 1.0),
                temperature=kwargs.get("temperature", 1.0),
                attention_mask=attention_mask,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=layershift(eoa, 0),
                pad_token_id=self.tokenizer.pad_token_id
            )
            model_outputs = self.process_interleaved_output(model_outputs)
            return model_outputs

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if current_input_text is not None:
                audio_tokens = torch.cat([layershift(current_audio_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                combined_input_ids = torch.cat([audio_tokens, current_input_text.unsqueeze(1)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values

            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            xt_logits = self.repetition_penalty(xt_logits, generated_ids[self.code_layer][:step], text_repetition_penalty)
            for i in range(self.code_layer):
                xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][:step], audio_repetition_penalty)

            if not text_end:
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)
            else:
                next_token_text = torch.tensor([pad_t], device=input_ids.device)
            
            next_tokens_audio = []
            for i in range(self.code_layer):
                if not audio_end and not decode_text_only and num_latency_tokens <= step:
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                else:
                    next_token_audio = torch.full((input_ids.size(0),), pad_a, device=input_ids.device)
                next_tokens_audio.append(next_token_audio)

            if eoa in next_tokens_audio or decode_text_only:
                audio_end = True
            if next_token_text == eot:
                text_end = True
            
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]
            generated_ids[self.code_layer][step] = next_token_text

            if self.model_config.use_text_stream:
                if audio_end and text_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break       
            else:
                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break     

        text_tokens = generated_ids[self.code_layer]
        generated_ids[self.code_layer] = text_tokens[: (text_tokens == eot).nonzero(as_tuple=True)[0][0]] if eot in text_tokens else text_tokens

        if eoa in generated_ids[self.code_layer - 1] and do_layershift:
            end_ids = (generated_ids[self.code_layer - 1] == eoa).nonzero(as_tuple=True)[0][0]
            for i in range(self.code_layer):
                audio_tokens = generated_ids[i]
                generated_ids[i] = audio_tokens[:end_ids]

        if upsampling_factor > 1:
            generated_ids[self.code_layer] = generated_ids[self.code_layer][::upsampling_factor]
            
        return generated_ids


    @torch.no_grad()
    def serial_generate(self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_values: Optional[List[torch.FloatTensor]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                **kwargs,
                ):
        kwargs["inference_mode"] = True

        inputs_embeds, attention_mask = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )
        max_new_tokens = kwargs.get("max_new_tokens", 360)
        generated_ids = [torch.zeros((max_new_tokens,), dtype=torch.long, device=input_ids.device) for _ in range(self.code_layer )]
        current_tokens = [None for _ in range(self.code_layer)]
        past_key_values = None

        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        
        text_repetition_penalty = kwargs.get("text_repetition_penalty", 1.0)
        audio_repetition_penalty = kwargs.get("audio_repetition_penalty", 1.0)
        decode_text_only = kwargs.get("decode_text_only", False)
        do_layershift = kwargs.get("do_layershift", True)
        if do_layershift:
            layershift = layer_shift
        else:
            layershift = simple_shift
        eot = self.model_config.vocab_config.eot
        eoa = self.model_config.vocab_config.eoa

        text_end = False
        audio_end = False

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            if None not in current_tokens:
                if text_end:
                    if current_tokens[0].item() == eot:
                        combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                        text_step = step
                    else:
                        combined_input_ids = torch.cat([layershift(current_tokens[i], i).unsqueeze(1) for i in range(self.code_layer)], dim=1)
                else:
                    combined_input_ids = torch.cat([ current_tokens[i].unsqueeze(1) for i in range(self.code_layer)], dim=1)
                if self.train_config.use_peft:
                    inputs_embeds = self.llm.model.model.embed_tokens(combined_input_ids)
                else:
                    inputs_embeds = self.llm.model.embed_tokens(combined_input_ids)
                inputs_embeds = torch.mean(inputs_embeds, dim=1).unsqueeze(1)
            
            outputs = self.llm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]
            past_key_values = outputs.past_key_values

            if not text_end:
                xt_logits = logits[..., :text_vocab_size]
                xt_logits = self.repetition_penalty(xt_logits, generated_ids[0][:step], text_repetition_penalty)
                next_token_text = self.sample_next_token(xt_logits[-1, :], **kwargs)

                if next_token_text == eot:
                    text_end = True
                
                for i in range(self.code_layer):
                    current_tokens[i] = next_token_text

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

                for i in range(self.code_layer):
                    generated_ids[i][step] = next_token_text
                
            else:
                if self.group_decode_adapter is not None:
                    xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                    xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
                else:
                    xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

                for i in range(self.code_layer):
                    xa_logits[i] = self.repetition_penalty(xa_logits[i], generated_ids[i][text_step:step], audio_repetition_penalty)

                next_tokens_audio = []
                for i in range(self.code_layer):
                    next_token_audio = self.sample_next_token(xa_logits[i][-1, :], **kwargs)
                    next_tokens_audio.append(next_token_audio)
                
                if eoa in next_tokens_audio or decode_text_only:
                    audio_end = True

                for i in range(self.code_layer):
                    current_tokens[i] = next_tokens_audio[i]

                attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)
               
                for i in range(self.code_layer):
                    generated_ids[i][step] = next_tokens_audio[i]

                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break            

        text_tokens = generated_ids[0]
        if eot in text_tokens:
            end_text_id = (text_tokens == eot).nonzero(as_tuple=True)[0][0]
            generated_ids.append(text_tokens[: end_text_id ] if eot in text_tokens else text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = generated_ids[i][ end_text_id+1: ]
        else:
            generated_ids.append(text_tokens)
            for i in range(self.code_layer):
                generated_ids[i] = []     

        return generated_ids
    
    @torch.no_grad()
    def sample_next_token(self, logits, **kwargs):
        """
        Generate the next token based on the model output logits.
        Supports both greedy decoding, top-k sampling, and top-p (nucleus) sampling.
        """
        do_sample = kwargs.get("do_sample", False)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 1.0)
        num_samples = kwargs.get("num_samples", 1)

        logits = logits.squeeze(0)
        logits = logits / temperature

        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            return torch.argmax(logits, dim=-1, keepdim=True)

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(-1, generated_ids.unsqueeze(0), score)

        return logits
        
    def extract_interleaved_tokens(self, preds, text_labels, preds_start_idx):
        """
        Extract predictions and labels in interleaved mode.
        """
        interleaved_text_num = self.train_config.interleaved_text_token_num
        interleaved_audio_num = self.train_config.interleaved_audio_token_num
        
        text_preds_batch = []
        text_labels_batch = []
        audio_preds_batch = []
        audio_labels_batch = []

        for i in range(preds.size(0)):
            text_preds = []
            text_labels_list = []
            audio_preds = []
            audio_labels = []

            start_idx = preds_start_idx[i].item()
            total_length = preds.size(1)
            idx = start_idx
            
            while idx < total_length:
                if idx + interleaved_text_num <= total_length:
                    text_preds.append(preds[i, idx:idx + interleaved_text_num].unsqueeze(0))
                    text_labels_list.append(text_labels[i, idx:idx + interleaved_text_num].unsqueeze(0))
                idx += interleaved_text_num
                
                if idx + interleaved_audio_num <= total_length:
                    audio_preds.append(preds[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                    audio_labels.append(text_labels[i, idx:idx + interleaved_audio_num].unsqueeze(0))
                idx += interleaved_audio_num
            
            text_preds_batch.append(torch.cat(text_preds, dim=1))
            text_labels_batch.append(torch.cat(text_labels_list, dim=1))
            audio_preds_batch.append(torch.cat(audio_preds, dim=1))
            audio_labels_batch.append(torch.cat(audio_labels, dim=1))

        text_preds = torch.cat(text_preds_batch, dim=0) if text_preds_batch else torch.empty(0, interleaved_text_num)
        text_labels = torch.cat(text_labels_batch, dim=0) if text_labels_batch else torch.empty(0, interleaved_text_num)
        audio_preds = torch.cat(audio_preds_batch, dim=0) if audio_preds_batch else torch.empty(0, interleaved_audio_num)
        audio_labels = torch.cat(audio_labels_batch, dim=0) if audio_labels_batch else torch.empty(0, interleaved_audio_num)
        
        return text_preds, text_labels, audio_preds, audio_labels
        
    def process_interleaved_output(self, model_outputs):
        """
        Parse the interleaved generation results and separate tokens into audio and text.
        """
        batch_size, seq_len = model_outputs.shape
        interleaved_audio_token_num = self.train_config.interleaved_audio_token_num
        interleaved_text_token_num = self.train_config.interleaved_text_token_num
        audio_shift = self.model_config.vocab_config.padded_text_vocabsize

        audio_tokens, text_tokens = [], []

        for i in range(batch_size):
            current_audio, current_text = [], []
            sequence = model_outputs[i]

            idx = 0
            while idx < seq_len:
                text_chunk = sequence[idx: idx + interleaved_text_token_num]
                current_text.append(text_chunk)
                idx += interleaved_text_token_num

                if idx < seq_len:
                    audio_chunk = sequence[idx: idx + interleaved_audio_token_num] - audio_shift
                    current_audio.append(audio_chunk)
                    idx += interleaved_audio_token_num

            audio_tokens.append(torch.cat(current_audio) if current_audio else torch.tensor([], device=model_outputs.device))
            text_tokens.append(torch.cat(current_text) if current_text else torch.tensor([], device=model_outputs.device))

        audio_tokens = torch.stack(audio_tokens) if audio_tokens else torch.empty((batch_size, 0), device=model_outputs.device)
        text_tokens = torch.stack(text_tokens) if text_tokens else torch.empty((batch_size, 0), device=model_outputs.device)

        return {
            "audio": audio_tokens,
            "text": text_tokens.squeeze(0),
        }
