import numpy as np
import pandas as pd
import argparse

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns


#############################################
# LOAD FILE
#############################################

def load_file(path):

    rows = []

    with open(path, encoding="utf8") as f:

        for line in f:

            line = line.strip()

            if line == "":
                continue

            parts = line.split("\t")

            if len(parts) != 9:
                continue

            # remove token column
            rows.append(parts[1:])

    return np.array(rows)


#############################################
# METRICS
#############################################

def compute_metrics(y_true, y_pred):

    acc = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    print("Accuracy :", acc)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1 Score :", f1)


#############################################
# CONFUSION MATRIX
#############################################

def plot_confusion(y_true, y_pred, title):

    labels = sorted(list(set(y_true) | set(y_pred)))

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10,8))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=labels,
        yticklabels=labels
    )

    plt.title(title + " Confusion Matrix")

    plt.ylabel("True")
    plt.xlabel("Predicted")

    plt.tight_layout()

    plt.show()


#############################################
# TAG FREQUENCY
#############################################

def plot_tag_frequency(y_true, y_pred, title):

    true_freq = pd.Series(y_true).value_counts()
    pred_freq = pd.Series(y_pred).value_counts()

    df = pd.DataFrame({
        "Gold": true_freq,
        "Predicted": pred_freq
    }).fillna(0)

    df.plot(kind="bar", figsize=(12,6))

    plt.title(title + " Tag Frequency")

    plt.ylabel("Count")

    plt.show()


#############################################
# EVALUATION
#############################################

def evaluate_column(gold, pred, name):

    print("\n"+"="*60)
    print("Evaluation for", name)
    print("="*60)

    compute_metrics(gold, pred)

    plot_confusion(gold, pred, name)

    plot_tag_frequency(gold, pred, name)


#############################################
# MAIN
#############################################

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--pred", required=True) #token, LCAT, Gender, Number, Person, Case, Vibhakti, POS, Chunk
    parser.add_argument("--gold", required=True) #token, LCAT, Gender, Number, Person, Case, Vibhakti, POS, Chunk

    args = parser.parse_args()

    pred = load_file(args.pred)
    gold = load_file(args.gold)

    n = min(len(pred), len(gold))

    pred = pred[:n]
    gold = gold[:n]

    columns = [
        "LCAT",
        "Gender",
        "Number",
        "Person",
        "Case",
        "Vibhakti",
        "POS",
        "Chunk"
    ]

    for i, name in enumerate(columns):

        evaluate_column(gold[:, i], pred[:, i], name)


if __name__ == "__main__":
    main()