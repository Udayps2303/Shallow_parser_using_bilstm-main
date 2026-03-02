import pickle

with open("char2index.pkl", "rb") as f:
    data = pickle.load(f)

print(data)