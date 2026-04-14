from gensim.models import Word2Vec
import os

# Read training data (tokenized sentences)
def load_sentences(file_path):
    sentences = []
    current_sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                token = line.split("\t")[0]  # first column = word
                current_sentence.append(token)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


# Load corpus
sentences = load_sentences("all.txt")

# Train Word2Vec model
model = Word2Vec(
    sentences,
    vector_size=200,   # must match your code (200)
    window=5,
    min_count=1,
    workers=4
)

# Save in Word2Vec binary format
model.wv.save_word2vec_format("word2vec.bin", binary=True)

print("Word2Vec file created: word2vec.bin")