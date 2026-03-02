import pickle
import argparse
from collections import defaultdict


def read_dataset(file_path):
    """
    Reads tab-separated file with format:
    token, pos, lcat, gender, number, person, case, vibhakti, chunk
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        sentence = []
        for line in f:
            line = line.strip()
            
            # Sentence boundary (empty line)
            if not line:
                if sentence:
                    data.append(sentence)
                    sentence = []
                continue
            
            parts = line.split('\t')
            if len(parts) != 9:
                continue  # skip malformed lines
            
            token, pos, lcat, gender, number, person, case, vibh, chunk = parts
            
            sentence.append({
                "token": token,
                "pos": pos,
                "lcat": lcat,
                "gender": gender,
                "number": number,
                "person": person,
                "case": case,
                "vibh": vibh,
                "chunk": chunk
            })
        
        if sentence:
            data.append(sentence)
    
    return data


def build_word2index(dataset):
    word_set = set()
    
    for sent in dataset:
        for item in sent:
            word_set.add(item["token"])
    
    word2index = {
        "<PAD>": 0,
        "<UNK>": 1
    }
    
    for i, word in enumerate(sorted(word_set), start=2):
        word2index[word] = i
    
    return word2index


def build_label_index(dataset, key):
    label_set = set()
    
    for sent in dataset:
        for item in sent:
            label_set.add(item[key])
    
    label2index = {}
    index2label = {}
    
    for idx, label in enumerate(sorted(label_set)):
        label2index[label] = idx
        index2label[idx] = label
    
    return label2index, index2label


def save_pickle(obj, filename):
    with open(filename, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved: {filename}")


def main(args):
    print("Reading dataset...")
    dataset = read_dataset(args.input_file)
    
    print(f"Total sentences: {len(dataset)}")
    
    # 1. Word2Index
    print("Building word2index...")
    word2index = build_word2index(dataset)
    save_pickle(word2index, "word2index.pkl")
    
    # 2. POS
    _, index2pos = build_label_index(dataset, "pos")
    save_pickle(index2pos, "index-to-pos-tb.pkl")
    
    # 3. Chunk
    _, index2chunk = build_label_index(dataset, "chunk")
    save_pickle(index2chunk, "index-to-chunk-tb.pkl")
    
    # 4. LCAT
    _, index2lcat = build_label_index(dataset, "lcat")
    save_pickle(index2lcat, "index-to-lcat-tb.pkl")
    
    # 5. Gender
    _, index2gender = build_label_index(dataset, "gender")
    save_pickle(index2gender, "index-to-gender-tb.pkl")
    
    # 6. Number
    _, index2number = build_label_index(dataset, "number")
    save_pickle(index2number, "index-to-number-tb.pkl")
    
    # 7. Person
    _, index2person = build_label_index(dataset, "person")
    save_pickle(index2person, "index-to-person-tb.pkl")
    
    # 8. Case
    _, index2case = build_label_index(dataset, "case")
    save_pickle(index2case, "index-to-case-tb.pkl")
    
    # 9. Vibhakti
    _, index2vibh = build_label_index(dataset, "vibh")
    save_pickle(index2vibh, "index-to-vibh-tb.pkl")
    
    print("\nAll PKL files generated successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create index PKL files from tab-separated NLP dataset")
    parser.add_argument("--input_file", type=str, required=True, help="Path to training txt file")
    
    args = parser.parse_args()
    main(args)