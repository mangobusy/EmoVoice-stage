from typing import List, Tuple


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def emotion_value_to_bin(value: float, num_bins: int) -> int:
    value = _clamp(value)
    if num_bins <= 1:
        return 0
    bin_id = int(value * num_bins)
    return min(bin_id, num_bins - 1)


def emotion_tokens_from_values(valence: float, arousal: float, vocab_config) -> List[int]:
    num_bins = vocab_config.emotion_bins # 100
    base = vocab_config.emotion_token_start # 152000
    valence_bin = emotion_value_to_bin(valence, num_bins)
    arousal_bin = emotion_value_to_bin(arousal, num_bins)
    return [base + valence_bin, base + num_bins + arousal_bin]


def emotion_values_from_tokens(tokens: List[int], vocab_config) -> Tuple[float, float]:
    num_bins = vocab_config.emotion_bins
    base = vocab_config.emotion_token_start
    valence_bin = tokens[0] - base
    arousal_bin = tokens[1] - base - num_bins
    valence = (valence_bin + 0.5) / num_bins
    arousal = (arousal_bin + 0.5) / num_bins
    return valence, arousal


def split_emotion_tokens(tokens: List[int], vocab_config) -> Tuple[List[int], List[int]]:
    num_bins = vocab_config.emotion_bins # 100
    base = vocab_config.emotion_token_start  # 152000
    emotion_end = base + num_bins * 2  # 152200
    if len(tokens) >= 2 and base <= tokens[0] < base + num_bins and base + num_bins <= tokens[1] < emotion_end:
        return tokens[:2], tokens[2:]
    return [], tokens