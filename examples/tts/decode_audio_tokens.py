import argparse
import json
import os
import sys
from typing import Iterable, List, Optional

import soundfile as sf
import torch

'''
python examples/tts/decode_audio_tokens.py \
  --jsonl /root/autodl-tmp/data/Data_preprocess/test_tokenizer.jsonl \
  --output-dir /root/autodl-tmp/data/Data_preprocess/test_tokenizer \
  --codec-decoder-path /root/autodl-tmp/EmoVoice/checkpoint/ckpts/CosyVoice/CosyVoice-300M-SFT \
  --audio-prompt-path /root/autodl-tmp/data/EmoVoice-DB-Raw/audio/neutral/gpt4o_6000_neutral_verse.wav \
  --token-field answer_cosyvoice_speech_token \
  --code-layer 1 \
  --num-latency-tokens 0 \
  --ensure-eoa
'''


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
sys.path.append(os.path.abspath(os.path.join(SCRIPT_DIR, "../../src")))

from tts_config import ModelConfig, TrainConfig
from utils.codec_utils import audio_decode_cosyvoice, setup_codec


def iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_num, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num}: {exc}") from exc


def build_codec(codec_decoder_path: str, cosyvoice_version: int, code_layer: int) -> tuple:
    train_config = TrainConfig(enable_ddp=False, enable_fsdp=False, enable_deepspeed=False)
    model_config = ModelConfig(codec_decoder_path=codec_decoder_path, cosyvoice_version=cosyvoice_version)
    model_config.vocab_config.code_layer = code_layer
    codec_decoder = setup_codec(train_config, model_config)
    return model_config, codec_decoder


def ensure_eoa(tokens: List[int], eoa: int, add_eoa: bool) -> List[int]:
    if eoa in tokens or not add_eoa:
        return tokens
    return tokens + [eoa]


def decode_tokens(
    tokens: List[int],
    model_config: ModelConfig,
    codec_decoder,
    audio_prompt_path: str,
    num_latency_tokens: int,
    code_layer: int,
    speed: float,
) -> Optional[torch.Tensor]:
    audio_tokens = [torch.tensor(tokens, dtype=torch.int32)]
    return audio_decode_cosyvoice(
        audio_tokens,
        model_config,
        codec_decoder,
        audio_prompt_path=audio_prompt_path,
        code_layer=code_layer,
        num_latency_tokens=num_latency_tokens,
        speed=speed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Decode CosyVoice audio tokens from a jsonl file.")
    parser.add_argument("--jsonl", required=True, help="Path to jsonl with audio tokens.")
    parser.add_argument("--token-field", default="answer_cosyvoice_speech_token", help="Field name for tokens.")
    parser.add_argument("--output-dir", required=True, help="Directory to save decoded wavs.")
    parser.add_argument("--codec-decoder-path", required=True, help="Path to CosyVoice decoder checkpoint directory.")
    parser.add_argument("--cosyvoice-version", type=int, default=1, choices=[1, 2], help="CosyVoice version.")
    parser.add_argument("--audio-prompt-path", required=True, help="Prompt wav used by CosyVoice decoder.")
    parser.add_argument("--code-layer", type=int, default=1, help="Number of codec layers in the token stream.")
    parser.add_argument("--num-latency-tokens", type=int, default=0, help="Latency tokens prepended to the stream.")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Output sample rate.")
    parser.add_argument("--speed", type=float, default=1.0, help="Speed factor for decoding.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional max items to decode.")
    parser.add_argument("--ensure-eoa", action="store_true", help="Append end-of-audio token if missing.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_config, codec_decoder = build_codec(
        codec_decoder_path=args.codec_decoder_path,
        cosyvoice_version=args.cosyvoice_version,
        code_layer=args.code_layer,
    )

    eoa = model_config.vocab_config.eoa

    for idx, item in enumerate(iter_jsonl(args.jsonl), start=1):
        if args.max_items is not None and idx > args.max_items:
            break
        key = item.get("key", f"item_{idx}")
        tokens = item.get(args.token_field)
        if tokens is None:
            raise ValueError(f"Missing token field '{args.token_field}' in item {key}.")
        tokens = ensure_eoa(tokens, eoa, args.ensure_eoa)
        audio_hat = decode_tokens(
            tokens,
            model_config,
            codec_decoder,
            audio_prompt_path=args.audio_prompt_path,
            num_latency_tokens=args.num_latency_tokens,
            code_layer=args.code_layer,
            speed=args.speed,
        )
        if audio_hat is None:
            raise ValueError(f"Failed to decode {key}: missing or early EOA.")
        out_path = os.path.join(args.output_dir, f"{key}.wav")
        sf.write(out_path, audio_hat.squeeze().cpu().numpy(), args.sample_rate)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()