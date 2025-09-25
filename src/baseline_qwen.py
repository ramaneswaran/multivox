# SPDX-License-Identifier: Apache-2.0
# Code credits: https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/qwen2_5_omni/only_thinker.py
"""
This example shows how to use vLLM for running offline inference 
with the correct prompt format on Qwen2.5-Omni (thinker only).
"""

import argparse
from typing import NamedTuple
import numpy as np
from decord import VideoReader, cpu
import vllm.envs as envs
from vllm import LLM, SamplingParams
from vllm.assets.audio import AudioAsset
from vllm.assets.image import ImageAsset
from vllm.assets.video import VideoAsset
from vllm.utils import FlexibleArgumentParser
import librosa
from pathlib import Path 
import json
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_video(video_path, max_frames_num):
    if type(video_path) == str:
        vr = VideoReader(video_path, ctx=cpu(0))
    else:
        vr = VideoReader(video_path[0], ctx=cpu(0))
    total_frame_num = len(vr)
    uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames  # (frames, height, width, channels)

class QueryResult(NamedTuple):
    inputs: dict
    limit_mm_per_prompt: dict[str, int]


# NOTE: The default `max_num_seqs` and `max_model_len` may result in OOM on
# lower-end GPUs.
# Unless specified, these settings have been tested to work on a single L4.

default_system = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba "
    "Group, capable of perceiving auditory and visual inputs, as well as "
    "generating text and speech.")

def get_use_audio_in_video_query(audio_path, video_path) -> QueryResult:
    question = "Think step by step and ground to audio and video before answering"
    prompt = (f"<|im_start|>system\n{default_system}<|im_end|>\n"
              "<|im_start|>user\n<|vision_bos|><|VIDEO|><|vision_eos|>"
              f"{question}<|im_end|>\n"
              f"<|im_start|>assistant\n")
    sampling_rate = 16000
    audio_and_sample_rate = (librosa.load(
                                    audio_path, 
                                    sr=sampling_rate)[0],
                                    sampling_rate
                            )

    video_data = load_video(video_path, 16)
    assert not envs.VLLM_USE_V1, ("V1 does not support use_audio_in_video. "
                                  "Please launch this example with "
                                  "`VLLM_USE_V1=0`.")
    return QueryResult(
        inputs={
            "prompt": prompt,
            "multi_modal_data": {
                "video": video_data,
                "audio": audio_and_sample_rate,
            },
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        },
        limit_mm_per_prompt={
            "audio": 1,
            "video": 1
        },
    )


def process_sample(sample):
    media_path = sample['input_vision']
    input_audio_path = sample['input_audio']
    
    return get_use_audio_in_video_query(input_audio_path, media_path)

def benchmark():
    parser = argparse.ArgumentParser(description="Inference on qwen")
    parser.add_argument('--input_path', 
                       required=True,
                       help='Path to the input file')
    args = parser.parse_args()
    input_path = Path(args.input_path)

    with open(input_path, 'r') as f:
        input_data = json.load(f)

    model_name = "Qwen/Qwen2.5-Omni-7B"

    sampling_params = SamplingParams(temperature=0.2, max_tokens=256)

    llm = LLM(model=model_name,
              max_model_len=12000,
              max_num_seqs=1,
              limit_mm_per_prompt={
                    "audio": 1,
                    "video": 1
                },
              seed=13)

    with ThreadPoolExecutor(max_workers=8) as executor:
        conversations = list(tqdm(executor.map(process_sample, input_data), total=len(input_data)))

    batch_size = 8
    for batch_start_idx in tqdm(range(0, len(conversations), batch_size)):
        batch_end_idx = batch_start_idx + batch_size 
        batch_convs = [_conv.inputs for _conv in conversations[batch_start_idx: batch_end_idx]]
        outputs = llm.generate(batch_convs,
                       sampling_params=sampling_params)
        batch_outputs = [output.outputs[0].text for output in outputs]
        for idx in range(batch_size):
            if (batch_start_idx + idx) < len(input_data):
                input_data[batch_start_idx + idx]['output_text'] = batch_outputs[idx]
    
        save_path = "output.json"
        with open(save_path, 'w') as f:
            json.dump(input_data, f, indent=2)

if __name__ == "__main__":
    benchmark()