import streamlit as st
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import logging
logging.basicConfig(level=logging.INFO)# OPTIONAL
# from PyDictionary import PyDictionary
from py_thesaurus import Thesaurus

# dictionary = PyDictionary()
# synonym = dictionary.synonym('mother')
# print("synonym:", synonym)
# st.stop()

from itertools import chain
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')

import datetime
import random

# synonyms = wordnet.synsets('change')
# lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
# lemmas

# hide sidebar
st.set_page_config(
        # layout="wide",
        page_title="GPTMinusOne",
        page_icon="🤖",
        initial_sidebar_state="collapsed",
        )



tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForMaskedLM.from_pretrained('bert-base-uncased')
model.eval()
# model.to('cuda')  # if you have gpu


def predict_masked_sent(text, top_k=20):
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


def randomly_doubled_spaces(text):
    text_list = text.split()
    text = ''
    braille_blank = '⠀'
    space = ' '
    for i in range(len(text_list)):
        text += text_list[i]
        # text += ' '
        text += space
        if random.random() < 0.05:
            # text += ' '
            text += space

    return text




st.title("GPTMinusOne")
st.markdown("### Obfuscate the use of AI")

context_size = st.sidebar.slider("Context Size", 0, 400, 80)

LOG_FILE = "log.csv"

# with open(LOG_FILE, "a") as f:
    # f.write(f'{datetime.datetime.now()}, None\n')

# input_text = st.text_input("Enter the text")
# input_text = st.text_area("Enter the text", height=200)
# input_text = st.text_area("Enter the text (Max length around 200 words)", height=200)
input_text = st.text_area("Enter the text", height=200)

with open(LOG_FILE, "a") as f:
    f.write(f'{datetime.datetime.now()}, {input_text[:30]}\n')

if not st.button('Obfuscate'):
    st.stop()

# if input_text:
    # st.stop()

new_text = input_text

num_words = len(input_text.split())

status_1 = st.empty()
with st.expander("Show surounding text"):
    text_old = st.empty()
    text_new = st.empty()
status_2 = st.empty()
status_bar = st.progress(0)

for i in range(num_words):
    status_bar.progress((i+1)/num_words)
    status_1.write(f'At word {i}/{num_words}')
    masked_text, original_word = add_mask_ad_index(new_text, i)
    # if "'" in original_word:
        # continue
    print(masked_text)
    first_half, second_half = masked_text.split("[MASK]")
    # num_chars = 20
    num_chars = context_size
    masked_text_window = first_half[-num_chars:] + "[MASK]" + second_half[:num_chars]
    print("masked_text_window:", masked_text_window)
    pred = predict_masked_sent(masked_text_window)
    print(pred)
    # synonyms = dictionary.synonym(original_word)
    # new_instance = Thesaurus(original_word)
    # synonyms = new_instance.get_synonym()

    # synonyms.extend(new_instance.get_synonym(pos='verb'))
    # synonyms.extend(new_instance.get_synonym(pos='adj'))


    synonyms_raw = wordnet.synsets(original_word)
    lemmas = set(chain.from_iterable([word.lemma_names() for word in synonyms_raw]))
    synonyms = list(lemmas)

    print("synonyms:", synonyms)
    # if pred[0][0] == original_word:
        # if pred[1][1] > 0.2:
            # new_text = replace_mask_for_word(masked_text, pred[1][0])
            # status_2.write("Replaced '%s' with '%s'"%(original_word, pred[1][0]))
            # status_2.write(f'At word {i}/{num_words}, replaced {original_word} with {pred[1][0]}')
    if not synonyms:
        continue

    for word, probability in pred:
        if word == original_word:
            continue
        if word in synonyms:
            if probability > 0.01:
                # check if word starts with capital letter
                if not original_word[0].isupper():
                    new_text = replace_mask_for_word(masked_text, word)
                    old_text_window = replace_mask_for_word(masked_text_window, original_word)
                    new_text_window = replace_mask_for_word(masked_text_window, word)
                    # text_old.write(f'Old: "{old_text_window}"')
                    text_old.write(f'Old: "{first_half[-num_chars:]} :red[{original_word}] {second_half[:num_chars]}"')
                    # text_new.write(f'New: "{new_text_window}"')
                    text_new.write(f'New: "{first_half[-num_chars:]} :green[{word}] {second_half[:num_chars]}"')

                    # status_2.write("Replaced '%s' with '%s'"%(original_word, word))
                    status_2.write(f'Replaced :red[{original_word}] with :green[{word}]')
                    break






    # status_1.write(f'At word {i}/{num_words}')





    # st.write("Masked word:", masked_text)
    # st.write("Predictions:", pred)
    # st.write("New text:", new_text)
    print("New text:", new_text)
    print("Predictions:", pred)
    print("Masked word:", masked_text)

    # st.write("")

new_text = randomly_doubled_spaces(new_text)


status_1.empty()
status_2.empty()
status_bar.empty()

# st.write("New text:")

# st.write(new_text)
st.text_area("Obfuscated text", new_text, height=400)


st.write('Feedback?')
st.markdown('<a href="mailto:tomdoerr96@gmail.com">Contact me!</a>', unsafe_allow_html=True)


