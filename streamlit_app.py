import streamlit as st
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

    return_list = []
    for i, pred_idx in enumerate(top_k_indices):
        predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
        token_weight = top_k_weights[i]
        print("[MASK]: '%s'"%predicted_token, " | weights:", float(token_weight))
        return_list.append((predicted_token, float(token_weight)))

    return return_list

def add_mask_ad_index(text, mask_index):
    text = text.split()
    original_word = text[mask_index]
    text[mask_index] = "[MASK]"
    return " ".join(text), original_word


def replace_mask_for_word(text, word):
    text = text.split()
    text[text.index("[MASK]")] = word
    return " ".join(text)



st.title("GPTMinusOne")

# input_text = st.text_input("Enter the text")
input_text = st.text_area("Enter the text")

new_text = input_text

num_words = len(input_text.split())

status_1 = st.empty()
status_2 = st.empty()

for i in range(num_words):
    masked_text, original_word = add_mask_ad_index(new_text, i)
    # if "'" in original_word:
        # continue
    print(masked_text)
    pred = predict_masked_sent(masked_text)
    print(pred)
    if pred[0][0] == original_word:
        if pred[1][1] > 0.15:
            new_text = replace_mask_for_word(masked_text, pred[1][0])
            status_2.write("Replaced %s with %s"%(original_word, pred[1][0]))
            # status_2.write(f'At word {i}/{num_words}, replaced {original_word} with {pred[1][0]}')

    status_1.write(f'At word {i}/{num_words}')





    # st.write("Masked word:", masked_text)
    # st.write("Predictions:", pred)
    # st.write("New text:", new_text)
    print("New text:", new_text)
    print("Predictions:", pred)
    print("Masked word:", masked_text)

    st.write("")


st.write("New text:")

st.text_area("", new_text, height=200)
st.write(new_text)

