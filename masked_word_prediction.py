#!/usr/bin/env python3

import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)# OPTIONAL



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# model.to('cuda')  # if you have gpu


def predict_masked_sent(text, top_k=5):
    # Tokenize input
    text = "[CLS] %s [SEP]"%text
    tokenized_text = tokenizer.tokenize(text)
    masked_index = tokenized_text.index("[MASK]")
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    # tokens_tensor = tokens_tensor.to('cuda')    # if you have gpu

    # Predict all tokens
    with torch.no_grad():
        outputs = model(tokens_tensor)
        predictions = outputs[0]

    probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
    top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))

        
predict_masked_sent("My [MASK] is so cute.", top_k=5)
'''
The above code will output:
[MASK]: 'mom'  | weights: 0.10288725048303604
[MASK]: 'brother'  | weights: 0.08429113030433655
[MASK]: 'dad'  | weights: 0.08260555565357208
[MASK]: 'girl'  | weights: 0.06902255117893219
[MASK]: 'sister'  | weights: 0.04804788902401924
'''
