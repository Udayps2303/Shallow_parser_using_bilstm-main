import numpy as np
import pickle
import fasttext
import argparse
import os


def load_word2index(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def create_memmap(word2index, fasttext_model_path, output_path, embedding_dim=300):
    print("Loading fastText model...")
    ft = fasttext.load_model(fasttext_model_path)

    vocab_size = len(word2index)
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")

    # Create memmap file
    embeddings = np.memmap(
        output_path,
        dtype='float32',
        mode='w+',
        shape=(vocab_size, embedding_dim)
    )

    print("Generating embeddings...")

    for word, idx in word2index.items():
        if word in ["<PAD>"]:
            embeddings[idx] = np.zeros(embedding_dim)
        else:
            embeddings[idx] = ft.get_word_vector(word)

    embeddings.flush()
    print(f"\nSaved embeddings.memmap at: {output_path}")

    # Save shape info
    with open("embeddings_shape.pkl", "wb") as f:
        pickle.dump((vocab_size, embedding_dim), f)

    print("Saved embeddings_shape.pkl")


def main(args):
    word2index = load_word2index(args.word2index)
    create_memmap(word2index, args.fasttext_model, args.output, args.dim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2index", required=True)
    parser.add_argument("--fasttext_model", required=True)
    parser.add_argument("--output", default="embeddings.memmap")
    parser.add_argument("--dim", type=int, default=300)

    args = parser.parse_args()
    main(args)