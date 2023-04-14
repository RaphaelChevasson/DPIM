from transformers import GPT2LMHeadModel, GPT2TokenizerFast
device = "cuda"
model_name = "distilgpt2"
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)
tokenizer = GPT2TokenizerFast.from_pretrained(model_name)


import torch
from tqdm import tqdm

def score(model, tokenizer, sentence):
    stride = 512  # must be equal to the model window size
    encodings = tokenizer(sentence, return_tensors="pt").to(device)

    max_length = model.config.n_positions
    nlls = []
    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs[0] * trg_len

        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl

print(score(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
print(score(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))
print(score(sentence='London is the capital of Great Britain. London is the cacaca of Great Britain. London is the capital of Great Britain.', stride=512, model=model, tokenizer=tokenizer))


from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch
import numpy as np

device = 'cuda'
model_name = 'bert-large-uncased'
model = AutoModelForMaskedLM.from_pretrained(model_name).to('cuda')
tokenizer = AutoTokenizer.from_pretrained(model_name)

def score(model, tokenizer, sentence):
    tensor_input = tokenizer.encode(sentence, return_tensors='pt').to(device)
    repeat_input = tensor_input.repeat(tensor_input.size(-1)-2, 1)
    mask = torch.ones(tensor_input.size(-1) - 1, device=device).diag(1)[:-2]
    masked_input = repeat_input.masked_fill(mask == 1, tokenizer.mask_token_id)
    labels = repeat_input.masked_fill( masked_input != tokenizer.mask_token_id, -100)
    with torch.no_grad():
        loss = model(masked_input, labels=labels).loss
    return np.exp(loss.item())

print(score(sentence='London is the capital of Great Britain.', model=model, tokenizer=tokenizer))
# 4.541251105675365
print(score(sentence='London is the capital of South America.', model=model, tokenizer=tokenizer))
# 6.162017238332462
print(score(sentence='London is the capital of Great Britain. London is the cacaca of Great Britain. London is the capital of Great Britain.', model=model, tokenizer=tokenizer))