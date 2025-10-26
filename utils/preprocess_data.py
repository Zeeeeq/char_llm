# preprocess_data.py
import os
import pickle

def preprocess_text(train_file, test_file=None):
    os.makedirs('data', exist_ok=True)

    with open(train_file, 'r') as f:
        text_train = f.read()

    with open(test_file, 'r') as f:
        text_test = f.read()

    # Build vocab
    full_text = text_train + text_test
    chars = sorted(list(set(full_text)))
    vocab_size = len(chars)

    ctoi = {ch: i for i, ch in enumerate(chars)}
    itoc = {i: ch for i, ch in enumerate(chars)}

    # Encode text
    train_data = [ctoi[c] for c in text_train]
    test_data = [ctoi[c] for c in text_test]

    # Save preprocessed files
    with open('data/encoded.pkl', 'wb') as f:
        pickle.dump({
            'train_data': train_data,
            'test_data': test_data,
            'ctoi': ctoi,
            'itoc': itoc,
            'vocab_size': vocab_size
        }, f)

if __name__ == "__main__":
    preprocess_text('data/text8_train.txt', 'data/text8_test.txt')
