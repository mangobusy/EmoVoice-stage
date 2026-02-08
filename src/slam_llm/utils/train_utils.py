# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import time
import yaml
from contextlib import nullcontext
from pathlib import Path
from pkg_resources import packaging

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
from tqdm import tqdm
from transformers import LlamaTokenizer

from slam_llm.utils.checkpoint_handler import (
    save_model_checkpoint, 
    save_model_and_optimizer_sharded, 
    save_optimizer_checkpoint, 
    save_model_checkpoint_peft,
    save_model_checkpoint_peft_full_shard
)
from slam_llm.policies import fpSixteen,bfSixteen_mixed, get_llama_wrapper
from slam_llm.utils.memory_utils import MemoryTrace

import wandb
import logging
logger = logging.getLogger(__name__)

def set_tokenizer_params(tokenizer: LlamaTokenizer):
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "left"

# Converting Bytes to Megabytes
def byte2mb(x):
    return int(x / 2**20)

def train(model, train_dataloader,eval_dataloader, tokenizer, optimizer, lr_scheduler, gradient_accumulation_steps, train_config, log_config, fsdp_config=None, local_rank=None, rank=None):
    """
    Trains the model on the given dataloader

    Args:
        model: The model to be trained
        train_dataloader: The dataloader containing the training data
        optimizer: The optimizer used for training
        lr_scheduler: The learning rate scheduler
        gradient_accumulation_steps: The number of steps to accumulate gradients before performing a backward/update operation
        num_epochs: The number of epochs to train for
        local_rank: The rank of the current node in a distributed setting
        train_config: The training configuration
        log_config: The logging configuration
        eval_dataloader: The dataloader containing the eval data
        tokenizer: tokenizer used in the eval for decoding the predicitons

    Returns: results dictionary containing average training and validation perplexity and loss
    """
    if train_config.use_fp16:
        scaler = torch.cuda.amp.GradScaler()
        if train_config.enable_fsdp:
            scaler = ShardedGradScaler()
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    
    train_prep = []
    train_loss = []
    train_acc = []
    train_audio_acc = []
    val_prep = []
    val_loss =[]
    val_text_acc = []
    val_audio_acc = []
    val_emotion_acc = []
    epoch_times = []
    checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    best_val_text_acc = 0.0
    best_val_audio_acc =0.0
    best_val_emotion_acc = 0.0
    stage = getattr(train_config, 'training_stage', 1)
    for epoch in range(train_config.num_epochs):
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            model.train()
            total_loss = 0.0
            total_acc = 0.0
            total_audio_acc = 0.0
            total_length = len(train_dataloader)//gradient_accumulation_steps
            # breakpoint()
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            for step, batch in enumerate(train_dataloader):
                for key in batch.keys():
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        batch[key] = batch[key].to(local_rank) if isinstance(batch[key], torch.Tensor) else batch[key]
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = batch[key][k2].to(local_rank) if isinstance(batch[key][k2], torch.Tensor) else batch[key][k2]
                    else:
                        batch[key] = batch[key].to('cuda:0') if isinstance(batch[key], torch.Tensor) else batch[key]
                        if isinstance(batch[key], dict):
                            for k2 in batch[key].keys():
                                batch[key][k2] = batch[key][k2].to('cuda:0') if isinstance(batch[key][k2], torch.Tensor) else batch[key][k2]
                with autocast():
                    outputs, *rest = model(**batch) # output(loss,logit,...)   *rest[text_acc, audio_acc, loss_recorder]
                    # print(*rest)                                                                     # loss_recorder[layer_loss, emotion_acc]
                    # breakpoint()                                                                         

                # ==================== 分阶段解包逻辑 START ====================
                if stage == 1:                    
                    if len(rest) > 2:
                        layer_loss = rest[2] # [mse, mae]
                        # 兼容处理：如果 rest[1] 是列表取第一个，如果是标量直接用
                        current_mae = rest[1][0] if isinstance(rest[1], list) else rest[1]
                        acc = current_mae # 这里用 MAE 代替 ACC 变量进行记录
                        audio_acc = [current_mae] # 保持格式一致
                    else:
                        # Fallback 防止越界
                        layer_loss = [outputs.loss.item(), 0.0]
                        acc = 0.0
                        audio_acc = [0.0]

               
                
                else:  # Stage 2: Speech Generation (Original)
                    acc = rest[0] if rest else -1 # text_acc
                    audio_acc = rest[1] if rest else -1   # audio acc

                    if train_config.modeling_paradigm == "parallel" or train_config.modeling_paradigm == "serial":
                        layer_loss = rest[2] if rest else -1
                    else:
                        layer_loss = [0]

                loss = outputs.loss
                # print(rest)
                # print("loss:", loss.item())
                # print(layer_loss)
                # breakpoint()
                loss = loss / gradient_accumulation_steps
                # 处理 layer_loss (List) 的平均
                emotion_acc = None
                val_acc = None
                aro_acc = None
                val_mae = None
                aro_mae = None
                V_loss = None
                aro_loss = None
                if isinstance(layer_loss, dict):
                    # emotion_acc = layer_loss.get("emotion_acc")
                    val_acc = layer_loss.get("val_acc")
                    aro_acc = layer_loss.get("aro_acc")
                    val_mae = layer_loss.get("val_mae")
                    aro_mae = layer_loss.get("aro_mae")
                    V_loss = layer_loss.get("val_loss")
                    aro_loss = layer_loss.get("aro_loss")
                    layer_loss_values = layer_loss.get("layer_loss")
                    if isinstance(layer_loss_values, list):
                        layer_loss_values = [l / gradient_accumulation_steps for l in layer_loss_values]
                        layer_loss["layer_loss"] = layer_loss_values
                    if emotion_acc is not None:
                        layer_loss["emotion_acc"] = emotion_acc / gradient_accumulation_steps
                    if val_acc is not None:
                        layer_loss["val_acc"] = val_acc / gradient_accumulation_steps
                    if aro_acc is not None:
                        layer_loss["aro_acc"] = aro_acc / gradient_accumulation_steps
                    if val_mae is not None:
                        layer_loss["val_mae"] = val_mae / gradient_accumulation_steps
                    if aro_mae is not None:
                        layer_loss["aro_mae"] = aro_mae / gradient_accumulation_steps
                    if V_loss is not None:
                        layer_loss["val_loss"] = V_loss / gradient_accumulation_steps
                    if aro_loss is not None:
                        layer_loss["aro_loss"] = aro_loss / gradient_accumulation_steps
                elif isinstance(layer_loss, list):
                    layer_loss = [l / gradient_accumulation_steps for l in layer_loss]
                # layer_loss = [l / gradient_accumulation_steps for l in layer_loss]
                # acc = acc / gradient_accumulation_steps
                # audio_acc = [acc / gradient_accumulation_steps for acc in audio_acc]
                acc = acc / gradient_accumulation_steps

                if isinstance(audio_acc, list):
                    audio_acc = [a / gradient_accumulation_steps for a in audio_acc]
                else:
                    audio_acc = audio_acc / gradient_accumulation_steps
                    audio_acc = [audio_acc] 
                # ==================== WandB 日志记录 START ====================
                if log_config.use_wandb and step % log_config.log_interval == 0:
                    log_dict = {}
                    
                    if stage == 1:
                        # Stage 1 Logs
                        log_dict["train_inner/loss"] = loss
                        log_dict["train_inner/emotion_mae"] = acc # 这里的 acc 实际上是 MAE
                        if len(layer_loss) >= 2:
                            log_dict["train_inner/emotion_mse_raw"] = layer_loss[0]
                            log_dict["train_inner/emotion_mae_raw"] = layer_loss[1]
                    else:
                        # Stage 2 Logs
                        log_dict["Total Loss"] = loss
                        log_dict["Text Accuracy"] = acc
                        for layer, a_acc in enumerate(audio_acc):
                            log_dict[f"Audio Accuracy (layer{layer})"] = a_acc

                        layer_loss_values = layer_loss.get("layer_loss") if isinstance(layer_loss, dict) else layer_loss
                        if isinstance(layer_loss_values, list) and len(layer_loss_values) > 1:
                            log_dict["Audio Loss"] = sum(layer_loss_values[:-1])
                            log_dict["Emotion Loss"] = layer_loss_values[-1]
                            
                            for layer, l in enumerate(layer_loss_values[:-1]):
                                log_dict[f"Audio Loss (layer{layer})"] = l
                        if isinstance(layer_loss, dict) and layer_loss.get("emotion_acc") is not None:
                            log_dict["Emotion Accuracy"] = layer_loss["emotion_acc"]
                        if isinstance(layer_loss, dict):
                            if layer_loss.get("emotion_acc") is not None:
                                log_dict["Emotion Accuracy"] = layer_loss["emotion_acc"]
                            if layer_loss.get("val_loss") is not None:
                                log_dict["Valence Loss"] = layer_loss["val_loss"]
                            if layer_loss.get("aro_loss") is not None:
                                log_dict["Arousal Loss"] = layer_loss["aro_loss"]
                            if layer_loss.get("val_acc") is not None:
                                log_dict["Valence Accuracy"] = layer_loss["val_acc"]
                            if layer_loss.get("aro_acc") is not None:
                                log_dict["Arousal Accuracy"] = layer_loss["aro_acc"]
                            if layer_loss.get("val_mae") is not None:
                                log_dict["Valence MAE"] = layer_loss["val_mae"]
                            if layer_loss.get("aro_mae") is not None:
                                log_dict["Arousal MAE"] = layer_loss["aro_mae"]


                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank == 0:
                            wandb.log(log_dict, step=(epoch * total_length + step))
                    else:
                        wandb.log(log_dict, step=(epoch * total_length + step))
                # ==================== WandB 日志记录 END ======================
                
                '''
                if log_config.use_wandb and step % log_config.log_interval == 0:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_text_accuracy":acc}, step=(epoch * total_length + step))
                            for layer, acc in enumerate(audio_acc):
                                wandb.log({f"train_inner/train_inner_audio_accuracy_layer{layer}":acc}, step=(epoch * total_length + step))
                            for layer, l in enumerate(layer_loss[:-1]):
                                wandb.log({f"train_inner/train_inner_audio_loss_layer{layer}":l}, step=(epoch * total_length + step))
                            wandb.log({f"train_inner/train_inner_text_loss":layer_loss[-1]}, step=(epoch * total_length + step))
                    else:
                        wandb.log({"train_inner/train_inner_loss":loss, "train_inner/train_inner_text_accuracy":acc}, step=(epoch * total_length + step))
                        for layer, acc in enumerate(audio_acc):
                            wandb.log({f"train_inner/train_inner_audio_accuracy_layer{layer}":acc}, step=(epoch * total_length + step))
                        for layer, l in enumerate(layer_loss[:-1]):
                            wandb.log({f"train_inner/train_inner_audio_loss_layer{layer}":l}, step=(epoch * total_length + step))
                        wandb.log({f"train_inner/train_inner_text_loss":layer_loss[-1]}, step=(epoch * total_length + step))
                    '''
                total_loss += loss.detach().float()
                total_acc += acc # text_acc
                # print("audio_acc:",audio_acc)
                # print("audio_acc[0]:",audio_acc[0])
                # print("audio_acc[1]:",audio_acc[1])
                # print("audio_acc[2]:",audio_acc[2])
                total_audio_acc = audio_acc[0]+audio_acc[1]+audio_acc[2]
                # total_audio_acc += audio_acc[0]
                if train_config.use_fp16:
                    # breakpoint()
                    # if fp16 is enabled, use gradient scaler to handle gradient update
                    scaler.scale(loss).backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        scaler.step(optimizer)
                        scaler.update()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"lr":current_lr}, step=(epoch * total_length + step))
                            else:
                                wandb.log({"lr":current_lr}, step=(epoch * total_length + step))
                        optimizer.zero_grad()
                        pbar.update(1)
                else:
                    # regular backpropagation when fp16 is not used
                    loss.backward()
                    if (step + 1) % gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                        optimizer.step()
                        if lr_scheduler is not None:
                            lr_scheduler.step()
                            current_lr = lr_scheduler.get_last_lr()[0]
                        else:
                            current_lr = optimizer.param_groups[0]["lr"]
                        if current_lr == 0:
                            break
                        if log_config.use_wandb and step % log_config.log_interval == 0:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    wandb.log({"lr":current_lr}, step=(epoch * total_length + step))
                            else:
                                wandb.log({"lr":current_lr}, step=(epoch * total_length + step))
                        optimizer.zero_grad()
                        pbar.update(1)
                # 更新进度条描述
                if stage == 1:
                    desc_str = f"Epoch: {epoch+1}, step {step}/{len(train_dataloader)} (loss: {loss.detach().float():.4f}, MAE: {acc:.4f})"
                else:
                    use_emotion_token_loss = getattr(train_config, "use_emotion_token_loss", False)
                    layer_loss_values = layer_loss.get("layer_loss") if isinstance(layer_loss, dict) else layer_loss
                    emotion_acc_value = None
                    if isinstance(layer_loss, dict):
                        emotion_acc_value = layer_loss.get("emotion_acc")
                    if use_emotion_token_loss and isinstance(layer_loss_values, list) and len(layer_loss_values) > 1:
                        audio_loss_value = sum(layer_loss_values[:-1])
                        emotion_loss_value = layer_loss_values[-1]
                        emotion_acc_value = emotion_acc_value if emotion_acc_value is not None else acc
                        if hasattr(audio_loss_value, "item"):
                            audio_loss_value = audio_loss_value.item()
                        if hasattr(emotion_loss_value, "item"):
                            emotion_loss_value = emotion_loss_value.item()
                        if hasattr(emotion_acc_value, "item"):
                            emotion_acc_value = emotion_acc_value.item()
                        desc_str = (
                            f"Epoch: {epoch+1}, step {step}/{len(train_dataloader)} "
                            f"(loss: {loss.detach().float():.4f}, audio_loss: {audio_loss_value:.4f}, "
                            f"emo_loss: {emotion_loss_value:.4f}, audio_acc: {audio_acc[0]:.4f}, "
                            f"text_acc: {acc:.4f},"
                            f"emo_acc: {emotion_acc_value:.4f})"
                        )
                # else:
                #     desc_str = f"Epoch: {epoch+1}, step {step}/{len(train_dataloader)} (loss: {loss.detach().float():.4f}, audio_acc: {audio_acc[0]:.4f}, text_acc: {acc:.4f})"
                # pbar.set_description(f"Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(train_dataloader)} completed (loss: {loss.detach().float():.4f}, audio_acc: {audio_acc[0]:.4f}, text_acc: {acc:.4f})")
                pbar.set_description(desc_str)

                if (epoch * total_length + step + 1) % train_config.validation_interval == 0 and train_config.run_validation:
                    eval_ppl, eval_loss, *rest = evaluation(model, train_config, eval_dataloader, local_rank, tokenizer)
                    # print("eval_ppl:",eval_ppl)
                    # print("eval_epoch_loss:",eval_epoch_loss)
                    # print("*rest:",*rest)
                    # ====================================================
                    if stage == 1:
                        # Stage 1: eval_acc 用作 MAE
                        eval_epoch_mae = rest[0] if rest else float("inf")
                        # 逻辑反转：对于 Stage 1，越小越好
                        is_best = eval_epoch_mae < best_val_acc if best_val_acc != 0 else True # best_val_acc 这里暂存 best_mae
                        current_metric = eval_epoch_mae
                    else:
                        eval_text_acc = rest[0] if rest else -1
                        eval_audio_acc = rest[1] if rest else -1
                        eval_emotion_acc = rest[2] if len(rest) > 2 else -1
                        eval_audio_loss = rest[3] if len(rest) > 3 else -1
                        eval_V_acc = rest[4] if len(rest) > 4 else -1
                        eval_A_acc = rest[5] if len(rest) > 5 else -1
                        eval_V_mae = rest[6] if len(rest) > 6 else -1
                        eval_A_mae = rest[7] if len(rest) > 7 else -1
                        eval_V_loss = rest[8] if len(rest) > 8 else -1
                        eval_A_loss = rest[9] if len(rest) > 9 else -1
                        is_best = eval_loss < best_val_loss # 或者使用 acc 判断
                        current_metric = eval_text_acc
                    # ====================================================
                    '''
                    eval_epoch_acc = rest[0] if rest else -1
                    eval_epoch_audio_acc = rest[1] if rest else -1'''
                    checkpoint_start_time = time.perf_counter()
                    
                    # if train_config.save_model and (eval_epoch_loss < best_val_loss):
                    # if train_config.save_model and (eval_epoch_loss < best_val_loss or eval_epoch_audio_acc > best_val_audio_acc or eval_epoch_acc > best_val_acc):
                    
                    
                    
                    
                    if train_config.save_model:
                        # ==================================================================================================================
                        '''checkpoint_name = f"{train_config.model_name}_epoch_{str(epoch+1)}_step_{step+1}" '''
                        # checkpoint_name = f"{train_config.model_name}_latest"  # 只保存最新的模型
                        checkpoint_name = f"model_{epoch+1}.pt"
                        # ==================================================================================================================

                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                        if train_config.use_peft:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"we are about to save the PEFT modules")
                            else:
                                logger.info(f"we are about to save the PEFT modules")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                            else:
                                logger.info(f"PEFT modules are saved in {train_config.output_dir} directory")
                        
                        elif not train_config.use_peft and train_config.freeze_llm:
                            logger.info(f"llm is frozen, we are about to save other parts.")
                            if train_config.enable_fsdp:
                                if fsdp_config.sharding_strategy == ShardingStrategy.FULL_SHARD:
                                    save_model_checkpoint_peft_full_shard(
                                            model, optimizer, rank, train_config, epoch=epoch
                                        )
                                elif fsdp_config.sharding_strategy == ShardingStrategy.NO_SHARD:
                                    if rank==0:
                                        save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                    dist.barrier()
                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )

                        else: #
                            if train_config.enable_fsdp:
                                if getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.FULL_STATE_DICT:
                                    save_model_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                elif getattr(StateDictType, fsdp_config.checkpoint_type) == StateDictType.SHARDED_STATE_DICT:
                                    logger.info(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                                    logger.info("=====================================================")

                                    save_model_and_optimizer_sharded(model, rank, train_config)
                                    if train_config.save_optimizer:
                                        save_model_and_optimizer_sharded(model, rank, train_config, optim=optimizer)
                                        logger.info(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                                        logger.info("=====================================================")

                                if train_config.save_optimizer:
                                    save_optimizer_checkpoint(
                                        model, optimizer, rank, train_config, epoch=epoch
                                    )
                                    logger.info(" Saving the FSDP model checkpoints and optimizer using FULL_STATE_DICT")
                                    logger.info("=====================================================")

                            elif train_config.enable_ddp:
                                if rank==0:
                                    save_model_checkpoint_peft(
                                            model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                        )
                                dist.barrier()
                                    
                            else:
                                save_model_checkpoint_peft(
                                        model, optimizer, rank, train_config, checkpoint_name=checkpoint_name
                                    )
                                
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            dist.barrier()
                    checkpoint_end_time = time.perf_counter() - checkpoint_start_time
                    checkpoint_times.append(checkpoint_end_time)
                    if eval_loss < best_val_loss:
                        best_val_loss = eval_loss
                        if train_config.enable_fsdp or train_config.enable_ddp:
                            if rank==0:
                                logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                        else:
                            logger.info(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                   
                    val_loss.append(eval_loss)
                    val_prep.append(eval_ppl)

                    if stage == 1:
                        # Stage 1 Metrics Recording
                        # 这里复用 best_val_acc 变量来存储 best_val_mae (越小越好)
                        if best_val_acc == 0.0 or eval_epoch_mae < best_val_acc:
                            best_val_acc = eval_epoch_mae
                            if rank == 0: logger.info(f"best eval MAE on epoch {epoch+1} is {best_val_acc}")
                        val_text_acc.append(eval_epoch_mae)
                        val_audio_acc.append(-1)
                        
                        if log_config.use_wandb and rank == 0:
                            wandb.log({
                                "valid/val_epoch_loss": eval_epoch_loss, 
                                "valid/val_mae": eval_epoch_mae, 
                                "valid/best_val_loss": best_val_loss,
                                "valid/best_val_mae": best_val_acc
                            })
                    else: # Stage 2 Metrics Recording
                        if rest:
                            if eval_text_acc > best_val_text_acc:
                                best_val_text_acc = eval_text_acc
                                if train_config.enable_fsdp or train_config.enable_ddp:
                                    if rank==0:
                                        logger.info(f"best eval text acc on epoch {epoch+1} is {best_val_text_acc}")
                                else:
                                    logger.info(f"best eval text acc on epoch {epoch+1} is {best_val_text_acc}")
                            val_text_acc.append(rest[0]) 

                            if eval_audio_acc > best_val_audio_acc:
                                best_val_audio_acc = eval_audio_acc
                                if train_config.enable_fsdp or train_config.enable_ddp:
                                    if rank==0:
                                        logger.info(f"best eval audio acc on epoch {epoch+1} is {best_val_audio_acc}")
                                else:
                                    logger.info(f"best eval audio acc on epoch {epoch+1} is {best_val_audio_acc}")
                            val_audio_acc.append(rest[1]) 

                            val_emotion_acc.append(eval_emotion_acc)
                            if eval_emotion_acc is not None and eval_emotion_acc > best_val_emotion_acc:
                                best_val_emotion_acc = eval_emotion_acc
                                if train_config.enable_fsdp or train_config.enable_ddp:
                                    if rank==0:
                                        logger.info(f"best eval emotion acc is {best_val_emotion_acc}")
                                else:
                                    logger.info(f"best eval emotion acc is {best_val_emotion_acc}")

                        else: 
                            val_text_acc.append(-1)
                            val_audio_acc.append(-1)
                            val_emotion_acc.append(-1)

                        if log_config.use_wandb:
                            if train_config.enable_fsdp or train_config.enable_ddp:
                                if rank==0:
                                    # wandb.log({"valid/val_loss":eval_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_text_acc":val_text_acc[-1], "valid/val_emotion_acc":val_emotion_acc[-1],"valid/val_audio_acc":val_audio_acc[-1], "valid/val_audio_loss":eval_audio_loss, "valid/val_best_audio_acc":best_val_audio_acc, "valid/val_best_text_acc":best_val_text_acc, "valid/val_best_emotion_acc":best_val_emotion_acc })
                                    wandb.log({"valid/val_loss":eval_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_text_acc":val_text_acc[-1], "valid/val_emotion_acc":val_emotion_acc[-1], "valid/val_audio_acc":val_audio_acc[-1], "valid/val_audio_loss":eval_audio_loss, "valid/val_valence_acc":eval_V_acc, "valid/val_arousal_acc":eval_A_acc, "valid/val_valence_mae":eval_V_mae, "valid/val_arousal_mae":eval_A_mae, "valid/val_valence_loss":eval_V_loss, "valid/val_arousal_loss":eval_A_loss, "valid/val_best_audio_acc":best_val_audio_acc, "valid/val_best_text_acc":best_val_text_acc, "valid/val_best_emotion_acc":best_val_emotion_acc })
                            else:
                                # wandb.log({"valid/val_loss":eval_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_text_acc":val_text_acc[-1], "valid/val_emotion_acc":val_emotion_acc[-1], "valid/val_audio_acc":val_audio_acc[-1], "valid/val_audio_loss":eval_audio_loss, "valid/val_best_audio_acc":best_val_audio_acc, "valid/val_best_text_acc":best_val_text_acc, "valid/val_best_emotion_acc":best_val_emotion_acc })
                                # wandb.log({"valid/val_loss":eval_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_text_acc":val_text_acc[-1], "valid/val_emotion_acc":val_emotion_acc[-1], "valid/val_audio_acc":val_audio_acc[-1], "valid/val_audio_loss":eval_audio_loss, "valid/val_valence_acc":eval_val_acc, "valid/val_arousal_acc":eval_aro_acc, "valid/val_valence_mae":eval_val_mae, "valid/val_arousal_mae":eval_aro_mae, "valid/val_valence_loss":eval_val_loss, "valid/val_arousal_loss":eval_aro_loss, "valid/val_best_audio_acc":best_val_audio_acc, "valid/val_best_text_acc":best_val_text_acc, "valid/val_best_emotion_acc":best_val_emotion_acc })
                                wandb.log({"valid/val_loss":eval_loss, "valid/val_perplexity":eval_ppl, "valid/best_val_loss":best_val_loss, "valid/val_text_acc":val_text_acc[-1], "valid/val_emotion_acc":val_emotion_acc[-1], "valid/val_audio_acc":val_audio_acc[-1], "valid/val_audio_loss":eval_audio_loss, "valid/val_valence_acc":eval_V_acc, "valid/val_arousal_acc":eval_A_acc, "valid/val_valence_mae":eval_V_mae, "valid/val_arousal_mae":eval_A_mae, "valid/val_valence_loss":eval_V_loss, "valid/val_arousal_loss":eval_A_loss, "valid/val_best_audio_acc":best_val_audio_acc, "valid/val_best_text_acc":best_val_text_acc, "valid/val_best_emotion_acc":best_val_emotion_acc })
                                
                if train_config.run_test_during_validation and stage != 1:
                    if train_config.enable_fsdp or train_config.enable_ddp:
                        if rank==0:
                            logger.info("=====================================")
                            logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                            with autocast():
                                logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                            logger.info("=====================================")
                        dist.barrier()
                    else:
                        logger.info("=====================================")
                        logger.info(f"Test the file {train_config.run_test_during_validation_file} during validation:")
                        with autocast():
                            logger.info(model.inference(train_config.run_test_during_validation_file, train_config.run_test_during_validation_prompt))
                        logger.info("=====================================")
            pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if torch.cuda.device_count() > 1 and (train_config.enable_fsdp or train_config.enable_ddp):

            # ============ 修复代码 START ============
            # dist.all_reduce 必须要 Tensor，如果是 float (Stage 1) 需要转换
            # 使用 local_rank 确保 Tensor 在正确的 GPU 上
            
            # 1. 转换 total_loss
            if not isinstance(total_loss, torch.Tensor):
                total_loss = torch.tensor(total_loss).to(local_rank)
            
            # 2. 转换 total_acc (报错点)
            if not isinstance(total_acc, torch.Tensor):
                total_acc = torch.tensor(total_acc).to(local_rank)
                
            # 3. 转换 total_audio_acc
            if not isinstance(total_audio_acc, torch.Tensor):
                total_audio_acc = torch.tensor(total_audio_acc).to(local_rank)
            # ============ 修复代码 END ============

            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_audio_acc, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(train_dataloader)
        train_epoch_acc = total_acc / len(train_dataloader)
        train_epoch_audio_acc = total_audio_acc / len(train_dataloader)
        if train_config.enable_fsdp or train_config.enable_ddp:
            train_epoch_loss = train_epoch_loss/world_size
            train_epoch_acc = train_epoch_acc/world_size
            train_epoch_audio_acc = train_epoch_audio_acc/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(train_perplexity)
        train_loss.append(train_epoch_loss)
        train_acc.append(train_epoch_acc)
        train_audio_acc.append(train_epoch_audio_acc)

        if log_config.use_wandb:
            if train_config.enable_fsdp or train_config.enable_ddp:
                if rank==0:
                    if stage == 1:
                        logger.info(f"Epoch {epoch+1}: loss={train_epoch_loss:.4f}, MAE={train_epoch_acc:.4f}, time {epoch_end_time}s")
                    else:
                        logger.info(f"Epoch {epoch+1}: ppl={train_perplexity:.4f}, loss={train_epoch_loss:.4f}, acc={train_epoch_acc:.4f}, time {epoch_end_time}s")
                    # wandb.log({"train/train_perplexity":train_perplexity, "train/train_epoch_loss":train_epoch_loss, "train/train_epoch_acc":train_epoch_acc, "train/train_epoch_audio_acc":train_epoch_audio_acc})
            else:
                wandb.log({"train_perplexity":train_perplexity, "train_epoch_loss":train_epoch_loss, "train_epoch_text_acc":train_epoch_acc, "train_epoch_audio_acc":train_epoch_audio_acc})

        if train_config.enable_fsdp or train_config.enable_ddp:
            if rank==0:
                logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")
        else:
            logger.info(f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s")

        if train_config.enable_fsdp:
            if rank==0:
                logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
                logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
                logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
                logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
                logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")
        else:
            logger.info(f"Max CUDA memory allocated was {memtrace.peak} GB")
            logger.info(f"Max CUDA memory reserved was {memtrace.max_reserved} GB")
            logger.info(f"Peak active CUDA memory was {memtrace.peak_active_gb} GB")
            logger.info(f"Cuda Malloc retires : {memtrace.cuda_malloc_retires}")
            logger.info(f"CPU Total Peak Memory consumed during the train (max): {memtrace.cpu_peaked + memtrace.cpu_begin} GB")

    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    avg_train_acc = sum(train_acc)/len(train_acc)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
        avg_eval_acc = sum(val_acc)/len(val_acc)
        avg_eval_audio_acc = sum(val_audio_acc)/len(val_audio_acc)
        avg_eval_emotion_acc = sum(val_emotion_acc)/len(val_emotion_acc)

    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    results['avg_train_acc'] = avg_train_acc
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
        results['avg_eval_acc'] = avg_eval_acc
        results['avg_eval_audio_acc'] = avg_eval_audio_acc
        results['avg_eval_emotion_acc'] = avg_eval_emotion_acc
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    return results

def evaluation(model,train_config, eval_dataloader, local_rank, tokenizer):
    """
    Evaluates the model on the given dataloader

    Args:
        model: The model to evaluate
        eval_dataloader: The dataloader containing the evaluation data
        local_rank: The rank of the current node in a distributed setting
        tokenizer: The tokenizer used to decode predictions

    Returns: eval_ppl, eval_epoch_loss
    """
    if train_config.enable_fsdp or train_config.enable_ddp:
        world_size = int(os.environ["WORLD_SIZE"])
    model.eval()
    eval_preds = []
    eval_loss = 0.0  # Initialize evaluation loss
    eval_text_acc = 0.0
    eval_audio_acc = 0.0
    eval_emotion_acc = 0.0
    eval_audio_loss = 0.0
    eval_val_acc = 0.0
    eval_aro_acc = 0.0
    eval_val_mae = 0.0
    eval_aro_mae = 0.0
    eval_val_loss = 0.0
    eval_aro_loss = 0.0
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext

    with MemoryTrace() as memtrace:
        total_length = len(eval_dataloader)
        pbar = tqdm(colour="green", desc=f"Evaluating Epoch", total=total_length, dynamic_ncols=True)
        for step, batch in enumerate(eval_dataloader):
            for key in batch.keys():
                if train_config.enable_fsdp or train_config.enable_ddp:
                    batch[key] = batch[key].to(local_rank) if isinstance(batch[key], torch.Tensor) else batch[key]
                else:
                    batch[key] = batch[key].to('cuda:0') if isinstance(batch[key], torch.Tensor) else batch[key]
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                # Forward pass and compute loss
                with autocast():
                    outputs, *rest = model(**batch)
                # acc = rest[0] if rest else -1 #text_acc
                # audio_acc = rest[1][0] if rest else -1   # seven layers of audio acc
                loss = outputs.loss
                eval_loss += loss.detach().float()
                
                if train_config.training_stage == 1:
                    # Stage 1 Evaluation (Regression)
                    # rest[1] 应该是 MAE (Scalar or List)
                    if len(rest) > 1:
                        mae_val = rest[1][0] if isinstance(rest[1], list) else rest[1]
                        eval_acc += mae_val # Accumulate MAE
                    # Stage 1 不需要解码 token，直接跳过 try-catch 部分
                
                else:
                    # Stage 2 Evaluation (Generation)
                    text_acc = rest[0] if rest else -1   # text_acc
                    audio_acc = rest[1] if rest else -1
                    layer_loss = rest[2] if len(rest) > 2 else None
                    # if isinstance(layer_loss, dict) and layer_loss.get("emotion_acc") is not None:
                    #     eval_emotion_acc += layer_loss["emotion_acc"]
                    if isinstance(layer_loss, dict):
                        if layer_loss.get("emotion_acc") is not None:
                            eval_emotion_acc += layer_loss["emotion_acc"]
                        if layer_loss.get("val_acc") is not None:
                            eval_val_acc += layer_loss["val_acc"]
                        if layer_loss.get("aro_acc") is not None:
                            eval_aro_acc += layer_loss["aro_acc"]
                        if layer_loss.get("val_mae") is not None:
                            eval_val_mae += layer_loss["val_mae"]
                        if layer_loss.get("aro_mae") is not None:
                            eval_aro_mae += layer_loss["aro_mae"]
                        if layer_loss.get("val_loss") is not None:
                            eval_val_loss += layer_loss["val_loss"]
                        if layer_loss.get("aro_loss") is not None:
                            eval_aro_loss += layer_loss["aro_loss"]
                    eval_text_acc += text_acc
                    eval_audio_acc += audio_acc[0]
                    layer_loss_values = layer_loss.get("layer_loss") if isinstance(layer_loss, dict) else layer_loss
                    if isinstance(layer_loss_values, list) and layer_loss_values:
                        eval_audio_loss += sum(layer_loss_values[:-1]) if len(layer_loss_values) > 1 else layer_loss_values[0]

                    # Decode predictions (仅在 Stage 2 进行)
                    try:
                        preds = torch.argmax(outputs.logits, -1)
                        eval_preds.extend(
                            tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
                        )
                    except Exception:
                        pass 
            pbar.update(1)
            if train_config.training_stage == 1:
                 pbar.set_description(f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_mae: {eval_acc/(step+1):.4f}")
            else:
                #  pbar.set_description(f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}")
                # pbar.set_description(
                #      f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, "
                #      f"eval_text_acc: {eval_text_acc/(step+1):.4f}, eval_audio_loss: {eval_audio_loss/(step+1):.4f}, "
                #      f"eval_emo_acc: {eval_emotion_acc:.4f}"
                #  )
                pbar.set_description(
                     f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, "
                     f"eval_text_acc: {eval_text_acc/(step+1):.4f}, eval_audio_loss: {eval_audio_loss/(step+1):.4f}, "
                     f"eval_emo_acc: {eval_emotion_acc:.4f}"
                     f"eval_val_mae: {eval_val_mae/(step+1):.4f}, eval_aro_mae: {eval_aro_mae/(step+1):.4f}"
                 )


                # eval_acc += acc
                # eval_audio_acc += audio_acc
            # Decode predictions and add to evaluation predictions list
            # try:
            #     preds = torch.argmax(outputs.logits, -1)
            #     eval_preds.extend(
            #         tokenizer.batch_decode(preds.detach().cpu().numpy(), skip_special_tokens=True)
            #     )
            # except Exception:
            #     pass  # vallex does not need to show it's result (we can't view any thing from abstract acoustic token)
            
            # pbar.update(1)
            # pbar.set_description(f"step: {step+1}/{total_length}, eval_loss: {eval_loss/(step+1):.4f}, eval_audio_acc: {eval_audio_acc/(step+1):.4f}, eval_acc: {eval_acc/(step+1):.4f}")

    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp or train_config.enable_ddp:

        # ============ 修复代码 START ============
        if not isinstance(eval_loss, torch.Tensor):
            eval_loss = torch.tensor(eval_loss).to(local_rank)
            
        if not isinstance(eval_text_acc, torch.Tensor):
            eval_text_acc = torch.tensor(eval_text_acc).to(local_rank)
            
        if not isinstance(eval_audio_acc, torch.Tensor):
            eval_audio_acc = torch.tensor(eval_audio_acc).to(local_rank)
        # if not isinstance(eval_emotion_acc, torch.Tensor):
        #     eval_emotion_acc = torch.tensor(eval_emotion_acc).to(local_rank)
        if not isinstance(eval_audio_loss, torch.Tensor):
            eval_audio_loss = torch.tensor(eval_audio_loss).to(local_rank)
        if not isinstance(eval_val_acc, torch.Tensor):
            eval_val_acc = torch.tensor(eval_val_acc).to(local_rank)
        if not isinstance(eval_aro_acc, torch.Tensor):
            eval_aro_acc = torch.tensor(eval_aro_acc).to(local_rank)
        if not isinstance(eval_val_mae, torch.Tensor):
            eval_val_mae = torch.tensor(eval_val_mae).to(local_rank)
        if not isinstance(eval_aro_mae, torch.Tensor):
            eval_aro_mae = torch.tensor(eval_aro_mae).to(local_rank)
        if not isinstance(eval_val_loss, torch.Tensor):
            eval_val_loss = torch.tensor(eval_val_loss).to(local_rank)
        if not isinstance(eval_aro_loss, torch.Tensor):
            eval_aro_loss = torch.tensor(eval_aro_loss).to(local_rank)
        # ============ 修复代码 END ============

        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_text_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_audio_acc, op=dist.ReduceOp.SUM)
        # dist.all_reduce(eval_emotion_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_audio_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_val_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_aro_acc, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_val_mae, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_aro_mae, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_val_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(eval_aro_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_loss = eval_loss / len(eval_dataloader)
    eval_text_acc = eval_text_acc / len(eval_dataloader)
    eval_audio_acc = eval_audio_acc / len(eval_dataloader)
    # eval_emotion_acc = eval_emotion_acc / len(eval_dataloader)
    eval_audio_loss = eval_audio_loss / len(eval_dataloader)
    eval_val_acc = eval_val_acc / len(eval_dataloader)
    eval_aro_acc = eval_aro_acc / len(eval_dataloader)
    eval_val_mae = eval_val_mae / len(eval_dataloader)
    eval_aro_mae = eval_aro_mae / len(eval_dataloader)
    eval_val_loss = eval_val_loss / len(eval_dataloader)
    eval_aro_loss = eval_aro_loss / len(eval_dataloader)
    
    if train_config.enable_fsdp or train_config.enable_ddp:
        eval_loss = eval_loss/world_size
        eval_text_acc = eval_text_acc/world_size
        eval_audio_acc = eval_audio_acc/world_size
        # eval_emotion_acc = eval_emotion_acc/world_size
        eval_audio_loss = eval_audio_loss/world_size
        eval_val_acc = eval_val_acc/world_size
        eval_aro_acc = eval_aro_acc/world_size
        eval_val_mae = eval_val_mae/world_size
        eval_aro_mae = eval_aro_mae/world_size
        eval_val_loss = eval_val_loss/world_size
        eval_aro_loss = eval_aro_loss/world_size

    eval_ppl = torch.exp(eval_loss)
    if eval_emotion_acc == 0.0 and (eval_val_acc != 0.0 or eval_aro_acc != 0.0):
        eval_emotion_acc = (eval_val_acc + eval_aro_acc) / 2

    # Print evaluation metrics
    if train_config.enable_fsdp or train_config.enable_ddp:
        if local_rank==0:
            logger.info(f" {eval_ppl=} {eval_loss=} {eval_text_acc=} {eval_audio_acc=} {eval_emotion_acc=}")
    else:
        logger.info(f" {eval_ppl=} {eval_loss=} {eval_text_acc=} {eval_audio_acc=} {eval_emotion_acc=}")

    return eval_ppl, eval_loss, eval_text_acc, eval_audio_acc, eval_emotion_acc, eval_audio_loss,eval_val_acc, eval_aro_acc, eval_val_mae, eval_aro_mae, eval_val_loss, eval_aro_loss

def freeze_transformer_layers(model, num_layer):
   for i, layer in enumerate(model.model.layers):
            if i < num_layer:
                for param in layer.parameters():
                    param.requires_grad = False


def check_frozen_layers_peft_model(model):
     for i, layer in enumerate(model.base_model.model.model.layers):
            for name, param in layer.named_parameters():
                logger.info(f"Layer {i}, parameter {name}: requires_grad = {param.requires_grad}")


def setup():
    """Initialize the process group for distributed training"""
    dist.init_process_group("nccl")


def setup_environ_flags(rank):
    """Set environment flags for debugging purposes"""
    os.environ["TORCH_SHOW_CPP_STACKTRACES"] = str(1)
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = str(1)
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # This flag will help with CUDA memory fragmentations that can lead into OOM in some cases.
    # Note this is only availble in PyTorch Nighlies (as of July 30 2023)
    # os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    if rank == 0:
        logger.info(f"--> Running with torch dist debug set to detail")


def cleanup():
    """Clean up the process group after training"""
    dist.destroy_process_group()


def clear_gpu_cache(rank=None):
    """Clear the GPU cache for all ranks"""
    if rank == 0:
        logger.info(f"Clearing GPU cache for all ranks")
    torch.cuda.empty_cache()


def get_parameter_dtypes(model):
    """Get the data types of model parameters"""
    parameter_dtypes = {}
    for name, parameter in model.named_parameters():
        parameter_dtypes[name] = parameter.dtype
    return parameter_dtypes

def print_model_size(model, config, rank: int = 0) -> None:
    """
    log model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"--> {config.model_name} has {total_params / 1e6} Million params\n")

def print_module_size(module, module_name, rank: int = 0) -> None:
    """
    Print module name, the number of trainable parameters and initialization time.

    Args:
        module: The PyTorch module.
        module_name (str): Name of the model.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        logger.info(f"--> Module {module_name}")
        total_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
        logger.info(f"--> {module_name} has {total_params / 1e6} Million params\n")


def get_policies(cfg, rank):
    """Get the policies for mixed precision and fsdp wrapping"""

    verify_bfloat_support = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and packaging.version.parse(torch.version.cuda).release >= (11, 0)
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )
    mixed_precision_policy = None
    wrapping_policy = None

    # Mixed precision
    if cfg.mixed_precision:
        bf16_ready = verify_bfloat_support

        if bf16_ready and not cfg.use_fp16:
            mixed_precision_policy = bfSixteen_mixed
            if rank == 0:
                logger.info(f"bFloat16 enabled for mixed precision - using bfSixteen policy")
        elif cfg.use_fp16:
            mixed_precision_policy = fpSixteen
            if rank == 0:
                logger.info(f"FP16 enabled")
        else:
            logger.info(f"bFloat16 support not present. Using FP32, and not mixed precision")
    wrapping_policy = get_llama_wrapper()
    return mixed_precision_policy, wrapping_policy

def save_train_params(train_config, fsdp_config, rank):
    """
    This function saves the train_config and FSDP config into a train_params.yaml.
    This will be used by converter script in the inference folder to fetch the HF model name or path.
    It also would be hepful as a log for future references.
    """
    # Convert the train_config and fsdp_config objects to dictionaries,
    # converting all values to strings to ensure they can be serialized into a YAML file
    train_config_dict = {k: str(v) for k, v in vars(train_config).items() if not k.startswith('__')}
    fsdp_config_dict = {k: str(v) for k, v in vars(fsdp_config).items() if not k.startswith('__')}
    # Merge the two dictionaries into one
    train_params_dict = {**train_config_dict, **fsdp_config_dict}
    # Construct the folder name (follwoing FSDP checkpointing style) using properties of the train_config object
    folder_name = (
    train_config.dist_checkpoint_root_folder
    + "/"
    + train_config.dist_checkpoint_folder
    + "-"
    + train_config.model_name
    )

    save_dir = Path.cwd() / folder_name
    # If the directory does not exist, create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # Convert the dictionary to a YAML string
    config_yaml = yaml.dump(train_params_dict, indent=4)
    file_name = os.path.join(save_dir,'train_params.yaml')

    # Check if there's a directory with the same name as the file
    if os.path.isdir(file_name):
        logger.info(f"Error: {file_name} is a directory, not a file.")
    else:
        # Write the YAML string to the file
        with open(file_name, 'w') as f:
            f.write(config_yaml)
        if rank==0:
            logger.info(f"training params are saved in {file_name}")
