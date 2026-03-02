import numpy as np
import pickle
import argparse
from tqdm import tqdm


def load_word2index(pkl_path):
    with open(pkl_path, 'rb') as f:
        word2index = pickle.load(f)
    return word2index


def load_pretrained_embeddings(embedding_file):
    """
    Loads embeddings from .vec file (fastText/GloVe format)
    """
    embeddings = {}
    
    with open(embedding_file, 'r', encoding='utf-8', errors='ignore') as f:
        first_line = f.readline()
        
        # Detect if first line is header
        if len(first_line.split()) == 2:
            pass  # skip header (fastText format)
        else:
            word, *vec = first_line.split()
            embeddings[word] = np.array(vec, dtype=np.float32)
        
        for line in tqdm(f, desc="Loading embeddings"):
            parts = line.rstrip().split(' ')
            if len(parts) < 10:
                continue
            word = parts[0]
            vector = np.asarray(parts[1:], dtype=np.float32)
            embeddings[word] = vector
    
    return embeddings


def create_memmap(word2index, pretrained_embeddings, output_file, embedding_dim=300):
    vocab_size = len(word2index)
    
    print(f"Vocab size: {vocab_size}")
    print(f"Embedding dim: {embedding_dim}")
    
    memmap = np.memmap(
        output_file,
        dtype='float32',
        mode='w+',
        shape=(vocab_size, embedding_dim)
    )
    
    # Initialize random for OOV words
    memmap[:] = np.random.uniform(-0.25, 0.25, (vocab_size, embedding_dim))
    
    found = 0
    
    for word, idx in tqdm(word2index.items(), desc="Building memmap"):
        if word in pretrained_embeddings:
            memmap[idx] = pretrained_embeddings[word]
            found += 1
    
    memmap.flush()
    
    print(f"\nFound embeddings for {found}/{vocab_size} words")
    print(f"Saved memmap to: {output_file}")


def main(args):
    print("Loading word2index...")
    word2index = load_word2index(args.word2index)
    
    print("Loading pretrained embeddings...")
    pretrained_embeddings = load_pretrained_embeddings(args.embedding_file)
    
    print("Creating embeddings.memmap...")
    create_memmap(
        word2index,
        pretrained_embeddings,
        args.output_file,
        args.dim
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--word2index", required=True, help="Path to word2index.pkl")
    parser.add_argument("--embedding_file", required=True, help=".vec embedding file")
    parser.add_argument("--output_file", default="embeddings.memmap", help="Output memmap file")
    parser.add_argument("--dim", type=int, default=300, help="Embedding dimension")
    
    args = parser.parse_args()
    main(args)