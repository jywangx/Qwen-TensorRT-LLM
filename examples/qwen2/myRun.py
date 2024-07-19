# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import ast
import csv
from pathlib import Path

import numpy as np
import torch
from utils.utils import (DEFAULT_HF_MODEL_DIRS, DEFAULT_PROMPT_TEMPLATES,
                   load_tokenizer, read_model_name, throttle_generator)

import tensorrt_llm
import tensorrt_llm.profiler
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS, ModelRunner
from default_config import default_config

import pdb
import json
from typing import List, Tuple
from tqdm import tqdm, trange
import random

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_output_len', type=int, required=False, default=512)
    parser.add_argument(
        '--max_attention_window_size',
        type=int,
        default=None,
        help=
        'The attention window size that controls the sliding window attention / cyclic kv cache behaviour'
    )
    parser.add_argument('--sink_token_length',
                        type=int,
                        default=None,
                        help='The sink token length.')
    parser.add_argument('--log_level', type=str, default='error')
    parser.add_argument('--engine_dir', type=str, default=default_config.engine_dir)
    parser.add_argument('--use_py_session',
                        default=False,
                        action='store_true',
                        help="Whether or not to use Python runtime session")
    parser.add_argument(
        '--model_type',
        type=str,
        choices=["chatml", "base"],
        help='Indicates whether the model is chatml or raw.',
        default='chatml')
    parser.add_argument(
        '--input_text',
        type=str,
        nargs='+',
        default=["你好，请问你叫什么？"])
    parser.add_argument(
        '--no_prompt_template',
        dest='use_prompt_template',
        default=True,
        action='store_false',
        help=
        "Whether or not to use default prompt template to wrap the input text.")
    parser.add_argument(
        '--input_file',
        type=str,
        help=
        'CSV or Numpy file containing tokenized input. Alternative to text input.',
        default=None)
    parser.add_argument(
        '--dataset_path',
        type=str,
        help=
        'dataset_path.',
        default=None)
    parser.add_argument('--max_input_length', type=int, default=default_config.max_input_len)
    parser.add_argument('--max_batch_size', type=int, default=default_config.trt_max_batch_size)
    parser.add_argument('--output_csv',
                        type=str,
                        help='CSV file where the tokenized output is stored.',
                        default=None)
    parser.add_argument('--output_npy',
                        type=str,
                        help='Numpy file where the tokenized output is stored.',
                        default=None)
    parser.add_argument(
        '--output_logits_npy',
        type=str,
        help=
        'Numpy file where the generation logits are stored. Use only when num_beams==1',
        default=None)
    parser.add_argument('--tokenizer_dir',
                        help="HF tokenizer config path",
                        default=default_config.tokenizer_dir)
    parser.add_argument(
        '--tokenizer_type',
        help=
        'Specify that argument when providing a .model file as the tokenizer_dir. '
        'It allows AutoTokenizer to instantiate the correct tokenizer type.')
    parser.add_argument('--vocab_file',
                        help="Used for sentencepiece tokenizers")
    parser.add_argument('--num_beams',
                        type=int,
                        help="Use beam search if num_beams >1",
                        default=1)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top_k', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.0)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--repetition_penalty', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--debug_mode',
                        default=False,
                        action='store_true',
                        help="Whether or not to turn on the debug mode")
    parser.add_argument('--no_add_special_tokens',
                        dest='add_special_tokens',
                        default=True,
                        action='store_false',
                        help="Whether or not to add special tokens")
    parser.add_argument('--streaming', default=False, action='store_true')
    parser.add_argument('--streaming_interval',
                        type=int,
                        help="How often to return tokens when streaming.",
                        default=5)
    parser.add_argument(
        '--prompt_table_path',
        type=str,
        help="Path to .npy file, exported by nemo_prompt_convert.py")
    parser.add_argument(
        '--prompt_tasks',
        help="Comma-separated list of tasks for prompt tuning, e.g., 0,3,1,0")
    parser.add_argument('--lora_dir',
                        type=str,
                        default=None,
                        nargs="+",
                        help="The directory of LoRA weights")
    parser.add_argument(
        '--lora_task_uids',
        type=str,
        default=None,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module")
    parser.add_argument('--lora_ckpt_source',
                        type=str,
                        default="hf",
                        choices=["hf", "nemo"],
                        help="The source of lora checkpoint.")
    parser.add_argument(
        '--num_prepend_vtokens',
        nargs="+",
        type=int,
        help="Number of (default) virtual tokens to prepend to each sentence."
        " For example, '--num_prepend_vtokens=10' will prepend the tokens"
        " [vocab_size, vocab_size + 1, ..., vocab_size + 9] to the sentence.")
    parser.add_argument(
        '--run_profiling',
        default=False,
        action='store_true',
        help="Run several 10 iterations to profile the inference latencies.")
    parser.add_argument(
        '--medusa_choices',
        type=str,
        default=None,
        help="Medusa choice to use, if not none, will use Medusa decoding."
        "   E.g.: [[0, 0, 0, 0], [0, 1, 0], [1, 0], [1, 1]] for 9 medusa tokens."
    )

    return parser.parse_args(args=args)


def sample_requests(tokenizer,
                dataset_path: str,
                num_requests: int,
                prompt_template=None,
                add_special_tokens=True,
                max_input_len=default_config.max_input_len,
                max_new_tokens=default_config.max_new_tokens,
                pad_id=None,
                model_name=None,
                model_version=None):
    if pad_id is None:
        pad_id = tokenizer.pad_token_id

    with open(dataset_path) as f:
        dataset = json.load(f)
    max_output_len = max_input_len + max_new_tokens
    # Filter out the conversations with less than 2 turns.
    dataset = [
        data for data in dataset
        if len(data["conversations"]) >= 2
    ]
    # Only keep the first two turns of each conversation.
    dataset = [
        (data["conversations"][0]["value"], data["conversations"][1]["value"])
        for data in dataset
    ]
    # pdb.set_trace()
    
    batch: List[str] = []
    tokenized_dataset = []
    for i in trange(500, desc="Tokenizing for sample"):
        prompt = dataset[i][0]
        output_text = dataset[i][1]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        raw_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompt_tokens = tokenizer([raw_text], return_tensors="pt").input_ids.squeeze()
        new_token_len = len(tokenizer(output_text).input_ids)
        tokenized_dataset.append((raw_text, prompt_tokens, new_token_len))
    # pdb.set_trace()

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, new_token_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or new_token_len < 4:
            # Prune too short sequences.
            continue
        if prompt_len > max_input_len or (prompt_len + new_token_len) > max_output_len:
            # Prune too long sequences.
            continue
        # limit by max_output_len 
        filtered_dataset.append((prompt, prompt_len, new_token_len))
    # pdb.set_trace()

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def print_output(tokenizer,
                 inference_result,
                 output_csv=None):
    
    csv_res = [['ID', 'Input_Len', 'Output_Len']]
    i = 0
    for res in inference_result:
        output_ids = res['output_ids']
        input_lengths = res['input_lengths']
        sequence_lengths = res['sequence_lengths']
        batch_size, _, _ = output_ids.size()
        for j in range(batch_size):
            inputs = output_ids[j][0][:input_lengths[j]].tolist()
            input_text = tokenizer.decode(inputs)
            print(f'Input [Text {i}] length {input_lengths[j]}')
            # print(f'Input [Text {i}] length {input_lengths}: \"{input_text}\"')
            output_begin = input_lengths[j]
            output_end = sequence_lengths[j][0]
            outputs = output_ids[j][0][
                output_begin:output_end
            ].tolist()
            output_text = tokenizer.decode(outputs)
            print(
                f'Output [Text {i} Beam {0}] length {output_end-output_begin}')
            # pdb.set_trace()
            csv_res.append([i, input_lengths[j], output_end.item()-output_begin])
            i += 1

    if output_csv is not None:
        output_file = Path(output_csv)
        output_file.parent.mkdir(exist_ok=True, parents=True)
        with open(output_file, 'w') as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            writer.writerows(csv_res)


def main(args):
    runtime_rank = tensorrt_llm.mpi_rank()
    logger.set_level(args.log_level)

    model_name, model_version = read_model_name(args.engine_dir)
    if args.tokenizer_dir is None:
        logger.warning(
            "tokenizer_dir is not specified. Try to infer from model_name, but this may be incorrect."
        )
        args.tokenizer_dir = DEFAULT_HF_MODEL_DIRS[model_name]

    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=args.tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    stop_words_list = None
    bad_words_list = None

    prompt_template = None
    if args.use_prompt_template and args.model_type =='chatml' and model_name in DEFAULT_PROMPT_TEMPLATES:
        prompt_template = DEFAULT_PROMPT_TEMPLATES[model_name]
    requests = sample_requests( tokenizer=tokenizer,
                                num_requests=5,
                                dataset_path=args.dataset_path,
                                prompt_template=prompt_template,
                                add_special_tokens=args.add_special_tokens,
                                max_input_len=args.max_input_length,
                                pad_id=pad_id,
                                model_name=model_name,
                                model_version=model_version )
    # pdb.set_trace()
    batch: List[str] = []
    input_ids_batchs = []
    max_new_tokens = 0

    runner_cls = ModelRunner
    runner_kwargs = dict(engine_dir=args.engine_dir,
                        lora_dir=args.lora_dir,
                        rank=runtime_rank,
                        debug_mode=args.debug_mode,
                        lora_ckpt_source=args.lora_ckpt_source)
    runner = runner_cls.from_dir(**runner_kwargs)
    
    inference_result = []
    for i, (prompt, prompt_len, new_token_len) in tqdm(enumerate(requests), total=len(requests), desc="prasing requests"):
        batch.append(prompt)
        max_new_tokens = max(max_new_tokens, new_token_len)
        if len(batch) < args.max_batch_size and i < len(requests) - 1:
            continue
        input_ids = []
        input_lengths = []
        for input_text in batch:
            input_id = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_input_length,
            ).input_ids.type(torch.int32)
            input_ids.append(input_id)
            input_lengths.append(input_id.shape[-1])
        # padding
        max_length = max(input_lengths)
        # do padding, should move outside the profiling to prevent the overhead
        for i in range(len(input_ids)):
            pad_size = max_length - input_lengths[i]
            pad = torch.ones([1, pad_size]).type(torch.int32) * tokenizer.pad_token_id
            input_ids[i] = torch.cat(
                [torch.IntTensor(input_ids[i]), pad],
                dim=-1
            )

        input_ids_batchs.append(dict(
            input_ids = torch.cat(input_ids, dim=0).cuda(),
            max_new_tokens = max_new_tokens
        ))

        batch = []
        max_new_tokens = 0

    for i, iib in tqdm(enumerate(input_ids_batchs), total=len(input_ids_batchs), desc="do inference"):
        input_ids = iib['input_ids']
        input_lengths = [x.size(0) for x in input_ids]
        logger.info(f'input_lengths: {input_lengths}')

        # pdb.set_trace()
        with torch.no_grad():
            outputs = runner.generate(
                input_ids,
                max_new_tokens=min(iib['max_new_tokens'], default_config.max_input_len + default_config.max_new_tokens - input_ids.shape[1]),
                max_attention_window_size=args.max_attention_window_size,
                sink_token_length=args.sink_token_length,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                repetition_penalty=args.repetition_penalty,
                presence_penalty=args.presence_penalty,
                frequency_penalty=args.frequency_penalty,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                lora_uids=args.lora_task_uids,
                prompt_table_path=args.prompt_table_path,
                prompt_tasks=args.prompt_tasks,
                streaming=args.streaming,
                output_sequence_lengths=True,
                return_dict=True,
                medusa_choices=args.medusa_choices)
            torch.cuda.synchronize()
        if runtime_rank == 0:
            output_ids = outputs['output_ids']
            sequence_lengths = outputs['sequence_lengths']
            context_logits = None
            generation_logits = None
            if runner.gather_context_logits:
                context_logits = outputs['context_logits']
            if runner.gather_generation_logits:
                generation_logits = outputs['generation_logits']
            inference_result.append(dict(
                output_ids = output_ids,
                input_lengths = input_lengths,
                sequence_lengths = sequence_lengths,
                context_logits = context_logits,
                generation_logits = generation_logits,
            ))
    
    if runtime_rank == 0:
        print_output(tokenizer, inference_result, output_csv=args.output_csv)


if __name__ == '__main__':
    args = parse_arguments()
    main(args)
