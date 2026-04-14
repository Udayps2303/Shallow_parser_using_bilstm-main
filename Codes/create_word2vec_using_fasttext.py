from gensim.models import FastText

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
                token = line.split("\t")[0]
                current_sentence.append(token)

    if current_sentence:
        sentences.append(current_sentence)

    return sentences


sentences = load_sentences("all.txt")

model = FastText(
    sentences,
    vector_size=200,
    window=5,
    min_count=1,
    workers=4
)

model.wv.save_word2vec_format("fasttext_word2vec.bin", binary=True)

print("FastText embeddings created")