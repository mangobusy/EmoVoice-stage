# emotion loss在total_text_vocab中算，解码时在emotion token范围内解码（Constraint Decoding）
import torch
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
        # print("input_ids shape:", input_ids.shape) # [btz, code_layer, seq_length]
        # print("input_ids:", input_ids)
        if input_ids is not None: 
            input_ids[input_ids == -1] = 0  
            if hasattr(self.llm.model, "embed_tokens"):
                # print("Using llm.model.embed_tokens") # yes
                inputs_embeds = self.llm.model.embed_tokens(input_ids)
            elif hasattr(self.llm.model.model, "embed_tokens"):
                # print("Using llm.model.model.embed_tokens")
                inputs_embeds = self.llm.model.model.embed_tokens(input_ids)
            else:
                # print("No embed_tokens found in llm model")
                inputs_embeds = self.llm.model.model.model.embed_tokens(input_ids)
        # print("inputs_embeds shape:", inputs_embeds.shape) # [btz, code_layer, seq_length, emb_dim]
        # print("inputs_embeds:", inputs_embeds)
        # breakpoint()
        if modality_mask is not None and encoder_outs is not None:
            # print("modality_mask shape:", modality_mask.shape)  
            # print("encoder_outs shape:", encoder_outs.shape)
            # print(self.train_config.modeling_paradigm)
            if self.train_config.modeling_paradigm == "parallel":
                modality_mask = modality_mask.unsqueeze(1).repeat(1, self.code_layer, 1)  # [btz, code_layer, seq_length]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=2)
                modality_lengths = torch.clamp(modality_mask.sum(dim=2), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    for j in range(self.code_layer):
                        start_idx = modality_mask_start_indices[i, j].item()
                        length = modality_lengths[i][j]
                        encoder_outs_pad[i, j, start_idx:start_idx+length] = encoder_outs[i, :length]
                
                inputs_embeds[:, :self.code_layer, :, :] = encoder_outs_pad[:, :self.code_layer, :, :] + inputs_embeds[:, :self.code_layer, :, :] * (~modality_mask[:, :, :, None])
        
                inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers

            elif self.train_config.modeling_paradigm == "interleaved":
                inputs_embeds = inputs_embeds.squeeze(1)  # [btz, seq_length, emb_dim]
                modality_mask_start_indices = (modality_mask == True).float().argmax(dim=1)
                modality_lengths = torch.clamp(modality_mask.sum(dim=1), max=encoder_outs.shape[1]).tolist()

                encoder_outs_pad = torch.zeros_like(inputs_embeds)
                for i in range(encoder_outs.shape[0]):
                    encoder_outs_pad[
                        i, modality_mask_start_indices[i]:modality_mask_start_indices[i]+modality_lengths[i]
                    ] = encoder_outs[i][:modality_lengths[i]]
                
                inputs_embeds = encoder_outs_pad + inputs_embeds * (~modality_mask[:, :, None])
            
            else:
                raise NotImplementedError
        
        inputs_embeds = torch.mean(inputs_embeds, dim=1)  # [btz, seq_length, emb_dim], average over the code layers
        # print("inputs_embeds after fusion shape:", inputs_embeds.shape) # [btz, seq_length, emb_dim]
        # print("inputs_embeds after fusion:", inputs_embeds)
        if kwargs.get("inference_mode", False):
            return inputs_embeds, attention_mask
        

        if self.train_config.modeling_paradigm == "serial":
            # print("labels shape:", labels.shape)  # [btz, code_layer, seq_length]
            # print("labels:", labels) # 把prompt和source text都标为-100，只保留了emotion token, answer text token和audio token
            temp_labels = labels[:,self.code_layer - 1] if labels is not None else None # 最后一个code layer
            # print("temp_labels shape:", temp_labels.shape) # [btz, seq_length]
            # print("temp_labels:", temp_labels)
            model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels= temp_labels)    # here we use the text token layer as the target label
            # print("model_outputs logits shape:", model_outputs.logits.shape)  # [btz, seq_length, vocab_size]
            # print("model_outputs logits:", model_outputs.logits)
            # print("model_outputs:", model_outputs)
            eot = self.model_config.vocab_config.eot
            text_pad_token = self.model_config.vocab_config.pad_t
            audio_pad_token = self.model_config.vocab_config.pad_a
            text_labels = torch.full_like(labels, -100) # [btz, code_layer, seq_length]
            audio_labels = torch.full_like(labels, -100) # [btz, code_layer, seq_length]
            batch_size, seq_size, length = labels.shape # [btz, code_layer, seq_length]
            

            for i in range(batch_size):
                eot_position = (labels[i, 0] == eot).nonzero(as_tuple=True)[0]  # audio和text分界线
                eot_pos = eot_position.item()  
                text_labels[i, :, :eot_pos+1] = labels[i, :, :eot_pos+1] # [btz, code_layer, seq_length] 只有prompt + source_text + emotion_token + answer_text部分有值，后面全是-100
                text_labels[i, :, eot_pos+1:] = text_pad_token  # 把-100改为pad token：151937
                audio_labels[i, :, eot_pos+1:] = labels[i, :, eot_pos+1:]  # [btz, code_layer, seq_length] 只有audio部分有值，前面全是-100
                ignore_pos = torch.where(labels[i, 0]!= -100)[0][0].item()  # emotion token前面全部ignore
                audio_labels[i, :, ignore_pos:eot_pos+1] = audio_pad_token
            text_labels = text_labels[:, 0, :] # 取第一个code layer 
            # print("text_labels shape:", text_labels.shape) 
            # print("text_labels:", text_labels)
        else:
            text_labels = labels[:,self.code_layer] if labels is not None else None
            audio_labels = labels[:, :self.code_layer] if labels is not None else None
            model_outputs = self.llm(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=text_labels)    # here we use the text token layer as the target label

        if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
            x_ori = model_outputs.logits # [btz, seq_length, vocab_size]
            text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize # 152200
            audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize # 4160
            xt = x_ori[..., :text_vocab_size]  # [btz, seq_length, text_vocab_size=152200]
            xa = []

            if self.group_decode_adapter is not None:
                x_audio_ori = x_ori[..., text_vocab_size:] # [btz, seq_length, audio_vocab_size=4160]
                x_audio = self.group_decode_adapter(x_audio_ori) # [btz, seq_length, audio_vocab_size*code_layer=4160*3=12480]
                for i in range(self.code_layer):
                    xa.append(x_audio[..., i * audio_vocab_size : (i + 1) * audio_vocab_size])
                    # print("xa[{}] shape:".format(i), xa[i].shape)
                    # xa=[btz, seq_length, audio_vocab_size=4160]*code_layer
            else:
                for i in range(self.code_layer):
                    xa.append(x_ori[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)])
                    # print("xa[{}] shape:".format(i), xa[i].shape)
            # breakpoint()
            loss_recorder = []
            total_loss, loss_recorder = self.compute_parallel_loss(xt, text_labels, xa, audio_labels)
            # print("total_loss:",total_loss)
            # print("loss_recorder:",loss_recorder)
            model_outputs.loss = total_loss
            # print("model_outputs.loss:",model_outputs.loss)

        elif self.train_config.modeling_paradigm == "interleaved":
            x_ori = model_outputs.logits
        else:
            raise NotImplementedError

        text_acc = -1
        audio_acc = [-1 for _ in range(self.code_layer)] if self.code_layer > 0 else -1
        if self.metric:
            with torch.no_grad():
                if self.train_config.modeling_paradigm == "parallel" or self.train_config.modeling_paradigm == "serial":
                    preds = torch.argmax(xt, -1)  # [1,132]
                    # text_acc = compute_accuracy(preds.detach()[:, :-1], text_labels.detach()[:, 1:], ignore_label=-100)
                    emotion_token_start = self.model_config.vocab_config.emotion_token_start
                    emotion_token_end = emotion_token_start + self.model_config.vocab_config.emotion_token_count
                    acc_labels = text_labels.detach().clone()
                    emotion_mask = (acc_labels >= emotion_token_start) & (acc_labels < emotion_token_end)
                    acc_labels[emotion_mask] = -100 # emotion token位置改成-100
                    text_acc = compute_accuracy(preds.detach()[:, :-1], acc_labels[:, 1:], ignore_label=-100)
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
                    
                    audio_acc = [ compute_accuracy(audio_preds.detach()[:, :-1], audio_labels.detach()[:, 1:], ignore_label=-100)]

                    loss_recorder = None
                else:
                    raise NotImplementedError
        # breakpoint()
        return model_outputs, text_acc, audio_acc, loss_recorder

    def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
        """
        Compute the parallel loss with separate metrics for Valence and Arousal.
        """
        text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
        audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
        layer_loss = [0 for _ in range(self.code_layer + 1)]
        
        use_emotion_token_loss = getattr(self.train_config, "use_emotion_token_loss", False)
        
        # 获取配置
        emotion_token_start = self.model_config.vocab_config.emotion_token_start
        emotion_token_count = self.model_config.vocab_config.emotion_token_count
        emotion_token_end = emotion_token_start + emotion_token_count
        # 假设 V 和 A 各占一半，或者你在 config 里定义了 emotion_bins
        # 如果是 V(100) + A(100)，那 emotion_bins = 100
        emotion_bins = getattr(self.model_config.vocab_config, "emotion_bins", emotion_token_count // 2)
        emotion_split = emotion_token_start + emotion_bins 
        
        # 初始化统计量 (全部设为 0)
        text_token_loss = torch.tensor(0.0, device=xt.device)
        emotion_loss = torch.tensor(0.0, device=xt.device)
        
        # 独立的指标
        V_loss = torch.tensor(0.0, device=xt.device)
        A_loss = torch.tensor(0.0, device=xt.device)
        V_acc = torch.tensor(0.0, device=xt.device)
        A_acc = torch.tensor(0.0, device=xt.device)
        V_mae = torch.tensor(0.0, device=xt.device) # Valence 平均误差
        A_mae = torch.tensor(0.0, device=xt.device) # Arousal 平均误差

        if text_labels is not None:
            # Flatten
            shifted_logits = xt[:, :-1, :].reshape(-1, text_vocab_size) 
            shifted_labels = text_labels[:, 1:].reshape(-1)             
            
            if use_emotion_token_loss:
                # 1. 计算全量 Element-wise Loss (不求和)
                raw_losses = F.cross_entropy(shifted_logits, shifted_labels, reduction='none', ignore_index=-100)
                
                # 2. 定义 Mask
                # 文本 Mask
                is_text = (shifted_labels >= 0) & (shifted_labels < emotion_token_start)
                # Valence Mask (Start ~ Split)
                is_valence = (shifted_labels >= emotion_token_start) & (shifted_labels < emotion_split)
                # Arousal Mask (Split ~ End)
                is_arousal = (shifted_labels >= emotion_split) & (shifted_labels < emotion_token_end)
                # 总 Emotion Mask
                is_emotion = is_valence | is_arousal
                
                # 3. 计算 Text Loss
                if is_text.any():
                    text_token_loss = raw_losses[is_text].mean()
                
                # 4. 计算 Emotion Loss (总)
                if is_emotion.any():
                    emotion_loss = raw_losses[is_emotion].mean()
                
                # 5. 分别计算 V 和 A 的指标
                with torch.no_grad():
                    # --- Valence Metrics ---
                    if is_valence.any():
                        # Loss
                        V_loss = raw_losses[is_valence].mean()
                        
                        # Predictions
                        v_preds = torch.argmax(shifted_logits[is_valence], dim=-1)
                        v_targets = shifted_labels[is_valence]
                        # diff = torch.abs(emotion_preds - emotion_targets)
                        # Top-1 Accuracy (Exact Match)
                        V_acc = (v_preds == v_targets).float().mean()
                        # Top-k Accuracy
                        # emotion_acc_at_1 = (diff <= 1).float().mean()
                        # emotion_acc_at_2 = (diff <= 2).float().mean()
                        # MAE (Mean Absolute Error) -> 更有价值
                        V_mae = torch.abs(v_preds.float() - v_targets.float()).mean()

                    # --- Arousal Metrics ---
                    if is_arousal.any():
                        # Loss
                        A_loss = raw_losses[is_arousal].mean()
                        
                        # Predictions
                        a_preds = torch.argmax(shifted_logits[is_arousal], dim=-1)
                        a_targets = shifted_labels[is_arousal]
                        
                        # Accuracy
                        A_acc = (a_preds == a_targets).float().mean()
                        
                        # MAE
                        A_mae = torch.abs(a_preds.float() - a_targets.float()).mean()

                layer_loss[self.code_layer] = emotion_loss

            else:
                emotion_loss = F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)
                layer_loss[self.code_layer] = emotion_loss
        else:
            emotion_loss = 0

        # [Audio Loss 计算保持不变]
        total_audio_loss = 0
        single_audio_loss = 0
        for i in range(self.code_layer):
            if audio_labels[:, i] is not None:
                single_audio_loss = F.cross_entropy(
                    xa[i][:, :-1, :].reshape(-1, audio_vocab_size), 
                    audio_labels[:, i, 1:].reshape(-1), 
                    ignore_index=-100
                )
                layer_loss[i] = single_audio_loss
                total_audio_loss += single_audio_loss

        # Return Logic
        if use_emotion_token_loss:
            emotion_weight = getattr(self.train_config, "emotion_token_loss_weight", 1.0)
            audio_weight = getattr(self.train_config, "audio_token_loss_weight", 1.0)
            text_weight = getattr(self.train_config, "text_token_loss_weight", 1.0)
            
            # 总 Loss 依然使用聚合的 emotion_loss (因为它是 V 和 A 的平均)
            # 你也可以改成 emotion_weight * (val_loss + aro_loss) / 2
            total_loss = (emotion_weight * emotion_loss) + \
                         (audio_weight * total_audio_loss) + \
                         (text_weight * text_token_loss)
            
            # 返回详细字典
            return total_loss, {
                "layer_loss": layer_loss, 
                "val_loss": V_loss,
                "aro_loss": A_loss,
                "val_acc": V_acc,
                "aro_acc": A_acc,
                "val_mae": V_mae,
                "aro_mae": A_mae
            } 
        else:
            total_loss = (emotion_loss + total_audio_loss) / (self.code_layer + 1)
            
        return total_loss, layer_loss


    # def compute_parallel_loss(self, xt, text_labels, xa, audio_labels):
    #     """
    #     Compute the parallel loss for text and audio layers.
    #     Refactored to ensure Global Competition between text and emotion tokens.
    #     """
    #     text_vocab_size = self.model_config.vocab_config.padded_text_vocabsize
    #     audio_vocab_size = self.model_config.vocab_config.padded_audio_vocabsize
    #     layer_loss = [0 for _ in range(self.code_layer + 1)]
        
    #     use_emotion_token_loss = getattr(self.train_config, "use_emotion_token_loss", False)
    #     emotion_token_start = self.model_config.vocab_config.emotion_token_start
    #     emotion_token_end = emotion_token_start + self.model_config.vocab_config.emotion_token_count
        
    #     # 初始化统计量
    #     emotion_acc = torch.tensor(0.0, device=xt.device)
    #     emotion_mae = torch.tensor(0.0, device=xt.device)
    #     text_token_loss = torch.tensor(0.0, device=xt.device)
    #     emotion_loss = torch.tensor(0.0, device=xt.device)

    #     if text_labels is not None:
    #         # [1] 准备数据：Flatten
    #         shifted_logits = xt[:, :-1, :].reshape(-1, text_vocab_size) # [N, 152200]
    #         shifted_labels = text_labels[:, 1:].reshape(-1)             # [N]
            
    #         if use_emotion_token_loss:
    #             # [2] 关键修改：计算全量 Loss，不进行 reduction (求平均)，保留每个 token 的 loss 值
    #             # 这样情感 Token 必须在 152200 个选项中胜出，才能降低 Loss
    #             raw_losses = F.cross_entropy(shifted_logits, shifted_labels, reduction='none', ignore_index=-100)
                
    #             # [3] 创建 Mask
    #             # 情感 Mask
    #             is_emotion = (shifted_labels >= emotion_token_start) & (shifted_labels < emotion_token_end)
    #             # 文本 Mask (排除 padding 和 情感 token)
    #             is_text = (shifted_labels >= 0) & (shifted_labels < emotion_token_start)
                
    #             # [4] 分离 Loss
    #             if is_text.any():
    #                 text_token_loss = raw_losses[is_text].mean()
                
    #             if is_emotion.any():
    #                 emotion_loss = raw_losses[is_emotion].mean()
                    
    #                 # [5] 计算准确率 (仅用于监控，不参与梯度)
    #                 with torch.no_grad():
    #                     # 获取情感位置的预测结果 (在全词表中取 argmax)
    #                     emotion_preds = torch.argmax(shifted_logits[is_emotion], dim=-1)
    #                     emotion_targets = shifted_labels[is_emotion]
                        
    #                     # 只有当模型在 152200 个词里准确选中了那个情感词，才算对
    #                     correct = (emotion_preds == emotion_targets).float()
    #                     emotion_acc = correct.mean()
                        
    #                     # B. 新增：平均绝对误差 (MAE)
    #                     # 这能告诉你模型偏离了多少个 Bin。如果是 100 bins，MAE=5.0 表示平均偏离 5%
    #                     diff = torch.abs(emotion_preds.float() - emotion_targets.float())
    #                     emotion_mae = diff.mean()

    #                     # C. (可选) 容差准确率 (Tolerance Accuracy)
    #                     # 只要误差在 5 个 bin 以内都算对
    #                     tolerance_correct = (diff <= 5).float().mean()
    #                     print(f"Tolerance-Acc: {tolerance_correct}") 

    #                     # 如果你想看 Valence/Arousal 单独的准确率，可以在这里加逻辑，
    #                     # 但 Loss 必须统一计算。
                
    #             # 记录 Loss 用于日志
    #             layer_loss[self.code_layer] = emotion_loss

    #         else:
    #             # 不区分情感，直接算总 Text Loss
    #             emotion_loss = F.cross_entropy(shifted_logits, shifted_labels, ignore_index=-100)
    #             layer_loss[self.code_layer] = emotion_loss
    #     else:
    #         emotion_loss = 0

    #     # [6] Audio Loss 计算 (保持原样)
    #     total_audio_loss = 0
    #     single_audio_loss = 0
    #     for i in range(self.code_layer):
    #         if audio_labels[:, i] is not None:
    #             single_audio_loss = F.cross_entropy(
    #                 xa[i][:, :-1, :].reshape(-1, audio_vocab_size), 
    #                 audio_labels[:, i, 1:].reshape(-1), 
    #                 ignore_index=-100
    #             )
    #             layer_loss[i] = single_audio_loss
    #             total_audio_loss += single_audio_loss

    #     # [7] Loss 加权组合
    #     if use_emotion_token_loss:
    #         emotion_weight = getattr(self.train_config, "emotion_token_loss_weight", 1.0)
    #         audio_weight = getattr(self.train_config, "audio_token_loss_weight", 1.0)
    #         text_weight = getattr(self.train_config, "text_token_loss_weight", 1.0)
            
    #         # 这里的 emotion_loss 已经是包含了"抑制文本Token"梯度的正确Loss了
    #         total_loss = (emotion_weight * emotion_loss) + \
    #                      (audio_weight * total_audio_loss) + \
    #                      (text_weight * text_token_loss)
            
    #         return total_loss, {"layer_loss": layer_loss, "emotion_acc": emotion_acc} 
    #     else:
    #         total_loss = (emotion_loss + total_audio_loss) / (self.code_layer + 1)
            
    #     return total_loss, layer_loss


    @torch.no_grad()
    def generate(self,
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
                inputs_embeds=inputs_embeds,                  # [btz, seq_len / 1, emb_dim]
                attention_mask=attention_mask,                # single sample, no need for attention mask
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits[0]                      # batch size is 1
            past_key_values = outputs.past_key_values       # Update past_key_values for the next step

            # Split logits into text and audio layers based on vocab size
            xt_logits = logits[..., :text_vocab_size]
            if self.group_decode_adapter is not None:
                xa_logits = self.group_decode_adapter(logits[..., text_vocab_size:])
                xa_logits = [xa_logits[..., i * audio_vocab_size : (i + 1) * audio_vocab_size] for i in range(self.code_layer)]
            else:
                xa_logits = [logits[..., text_vocab_size + audio_vocab_size * i : text_vocab_size + audio_vocab_size * (i + 1)] for i in range(self.code_layer)]

            # Apply repetition penalty to the logits
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
            
            # Update input_ids for the next step
            current_input_text = next_token_text
            for i in range(self.code_layer):
                current_audio_tokens[i] = next_tokens_audio[i]

            attention_mask = torch.cat([attention_mask, torch.ones((input_ids.size(0), 1), device=input_ids.device)], dim=1)

            # Append generated tokens to the tensor
            for i in range(self.code_layer):
                generated_ids[i][step] = next_tokens_audio[i]  # Audio layers
            generated_ids[self.code_layer][step] = next_token_text  # Text layer

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

        # Concatenate the generated tokens to form the complete sequence
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
        
        # --- [Modification 1] 获取情感词表配置 ---
        emotion_token_start = self.model_config.vocab_config.emotion_token_start
        emotion_token_count = self.model_config.vocab_config.emotion_token_count
        emotion_token_end = emotion_token_start + emotion_token_count
        # ----------------------------------------
        
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
        
        # 标记是否已经生成了情感token (假设每个序列只需要生成一次，且在最开始)
        emotion_generated = False 

        for step in tqdm(range(max_new_tokens), desc="Generating"):
            # ... (这部分 Input Embedding 构建逻辑保持不变) ...
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
                # --- [Modification 2 & 3] 情感 Token 的强制推理逻辑 ---
                
                # 假设 step 0 是必须生成 Emotion 的位置
                # 如果你的 prompt 已经包含了 emotion，这里需要调整条件
                is_emotion_step = (step == 0) 

                if is_emotion_step:
                    # [关键点]：如果是情感步，我们需要看 Emotion 范围的 Logits，而不是只看 Text
                    # 先取出足够覆盖 Emotion 区域的 Logits
                    # 假设 emotion id 在 text id 后面
                    full_logits = logits[..., :emotion_token_end] 
                    
                    # [关键点]：Vocabulary Masking (Mask 掉所有非情感 Token)
                    mask = torch.full_like(full_logits, float('-inf'))
                    # 只保留 emotion 区域为 0 (即保留原始 Logits)
                    mask[..., emotion_token_start:emotion_token_end] = 0
                    
                    # 应用 Mask
                    masked_logits = full_logits + mask
                    
                    # 采样
                    # 注意：repetition penalty 在第一步通常不需要，或者仅针对 emotion 范围
                    # 这里为了安全直接采样
                    next_token_text = self.sample_next_token(masked_logits[-1, :], **kwargs)
                    
                    emotion_generated = True # 标记完成
                    
                else:
                    # --- 标准文本生成 ---
                    # 此时只看 Text 范围
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
                # ... (Audio 生成部分保持不变) ...
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
                    generated_ids[i][step] = next_tokens_audio[i]  # Audio layers

                if audio_end:
                    for i in range(self.code_layer):
                        generated_ids[i] = generated_ids[i][:step+1]
                    break            

        # ... (后续处理代码保持不变) ...
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

        # Adjust logits with temperature
        logits = logits.squeeze(0)
        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Make sure top_k is within the vocab size
            values, indices = torch.topk(logits, top_k)
            logits[logits < values[..., [-1]]] = -float('Inf')  # Filter tokens not in top_k

        # Top-p filtering (nucleus sampling)
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        if do_sample:
            # Perform sampling
            return torch.multinomial(F.softmax(logits, dim=-1), num_samples=num_samples)
        else:
            # Greedy decoding (argmax)
            return torch.argmax(logits, dim=-1, keepdim=True)

    def repetition_penalty(self, logits, generated_ids, repetition_penalty):
        """
        Apply repetition penalty to the logits.
        """
        if repetition_penalty == 1.0:
            return logits

        # Gather the logits for generated_ids
        score = torch.gather(logits, -1, generated_ids.unsqueeze(0))

        # Apply penalty
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)

        # Scatter the updated scores back into logits
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