import time
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

from customized_gpt2 import CustomizedGPT2LMHeadModel

@torch.no_grad()
def customized_greedy_decoding(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        outputs = custom_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch['input_ids'] = torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1)
        tokenized_batch['attention_mask'] = torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1)

        res = torch.cat([res, output_tokens], dim=-1)

    return res, time.time() - start_time

def prefix_caching_decoding(batch):
    prefix = "Once upon a time"
    input_ids = tokenizer.encode(prefix, return_tensors="pt")

    # 缓存前缀的中间状态
    with torch.no_grad():
        outputs = custom_model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values  # 缓存前缀的注意力键值对

    # 生成后续 token
    generated_tokens = []
    for timestep in range(MAX_NEW_LENGTH):# 生成 10 个 token
        next_token_logits = outputs.logits[:, -1, :]  # 取最后一个 token 的 logits
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)  # 选择概率最高的 token
        generated_tokens.append(next_token.item())

        # 使用缓存的 past_key_values 生成下一个 token
        outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
        past_key_values = outputs.past_key_values  # 更新缓存

    # 解码生成的 token
    generated_text = tokenizer.decode(generated_tokens)
    print(f"Generated Text: {generated_text}")

    return res, time.time() - start_time

@torch.no_grad()
def golden_greedy_decoding_wo_cache(batch):
    tokenized_batch = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=128).to('cuda')
    res = tokenized_batch['input_ids']
    start_time = time.time()
    for timestep in range(MAX_NEW_LENGTH):
        tokenized_batch = original_model.prepare_inputs_for_generation(**tokenized_batch)
        outputs = original_model(**tokenized_batch)
        output_tokens = torch.argmax(outputs['logits'][:,-1], dim=-1, keepdim=True)
        tokenized_batch = {
            "input_ids": torch.cat([tokenized_batch['input_ids'], output_tokens], dim=-1),
            "attention_mask": torch.cat([tokenized_batch['attention_mask'], torch.ones_like(output_tokens)], dim=-1),
        }
        res = torch.cat([res, output_tokens], dim=-1)
    
    return res, time.time() - start_time


if __name__ == "__main__":
    MAX_NEW_LENGTH = 100
    bsz = 16
    times = [0, 0]

    print(torch.cuda.is_available())
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    original_model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map='cuda')
    custom_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")
    prefix_caching_model = CustomizedGPT2LMHeadModel.from_pretrained("openai-community/gpt2", attn_implementation="eager", device_map="cuda")

    with open("data.txt") as f:
        prompt_dataset = [i.strip() for i in f.readlines()]

    for i in range(0, (len(prompt_dataset) + bsz - 1) // bsz):
        batch = prompt_dataset[i * bsz: (i + 1) * bsz]
        golden_wo_cache_res, golden_wo_cache_time = golden_greedy_decoding_wo_cache(batch)
        custom_res, custom_time = customized_greedy_decoding(batch)

        times[0] += golden_wo_cache_time
        times[1] += custom_time

        assert torch.equal(golden_wo_cache_res, custom_res), "Decoding results are not equal"

    print("Time taken for golden greedy decoding without KV cache: ", times[0])
    print("Time taken for customized greedy decoding: ", times[1])
