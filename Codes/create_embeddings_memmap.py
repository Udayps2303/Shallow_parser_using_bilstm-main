import numpy as np
import pickle
import argparse


def load_word2index(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def create_random_embeddings(word2index, output_path, embedding_dim=200):
    
    vocab_size = len(word2index)

    print("Vocabulary Size:", vocab_size)
    print("Embedding Dimension:", embedding_dim)

    # Create memmap file
    embeddings = np.memmap(
        output_path,
        dtype="float32",
        mode="w+",
        shape=(vocab_size, embedding_dim)
    )

    print("Generating random embeddings...")

    for word, idx in word2index.items():

        if word == "<PAD>":
            embeddings[idx] = np.zeros(embedding_dim)
        else:
            embeddings[idx] = np.random.uniform(-0.25, 0.25, embedding_dim)

    embeddings.flush()

    print("Embeddings saved to:", output_path)


def main(args):
    word2index = load_word2index(args.word2index)
    create_random_embeddings(word2index, args.output, args.dim)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--word2index", required=True)
    parser.add_argument("--output", default="embeddings.memmap")
    parser.add_argument("--dim", type=int, default=200)

    args = parser.parse_args()

    main(args)