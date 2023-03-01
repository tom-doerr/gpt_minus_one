from ubuntu:20.04

run apt update && apt install -y python3 python3-pip

# git
#run apt install -y git
run apt update && apt install -y git

run pip install \
tqdm \
streamlit

run pip install streamlit-authenticator
#run pip install openai
#run pip install streamlit-chat
#run pip install streamlit-javascript

run pip install stability-sdk

run pip install rembg

run pip install -U protobuf

# install cargo
#run apt install -y curl
run apt update && apt install -y curl
run curl https://sh.rustup.rs -sSf | sh -s -- -y
run /root/.cargo/bin/cargo install cargo-watch

# add cargo to path
env PATH="/root/.cargo/bin:${PATH}"

run cargo install vtracer

run pip install psutil

run pip install git+https://github.com/hunkim/streamlit-google-oauth

run pip install torch
run pip install transformers

run pip install PyDictionary
run pip install py-thesaurus
run pip install nltk
