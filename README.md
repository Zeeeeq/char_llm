# char_llm
Project on next-character prediction using Decoder-Only Transformer

## Goal Of This Project
Implement and train a small transformer that learns the
conditional distribution of the next character given a fixed-length
context window of the preceding characters.

This project explores transformer architecture
design, training dynamics, and hyperparameter tuning at a manageable
scale. Through systematic experimentation, we will identify an effective
configuration for the final model.

## Install

```sh
pip install -r requirements.txt
```

## Data Preperation

```sh
python util/preprocess_data.py
```


