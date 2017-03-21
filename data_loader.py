#!/usr/bin/env python

import sys
import numpy as np
import pandas as pd

import audio_processor as ap

def load_data():
    labels = load_labeles()
    examples = load_mel_examples()

    t = (labels, examples)
    print(t)
    return t

def load_labeles(limit=10, fname="./data/y_labels.tsv"):
    df = pd.read_csv(fname, delimiter="\t")
    df = df.iloc[:, 1:][:limit]
    print(df)
    print(df.as_matrix())
    return df.as_matrix()

def load_mel_examples(limit=10, fname="./data/clip_info_final.csv", num_rows=96, num_cols=1366):
    df = pd.read_csv(fname, delimiter="\t")
    print(df.info())
    mp3_paths = df["mp3_path"][:limit]

    examples = np.empty((0, 1, num_rows, num_cols), dtype=float)
    for i, p in enumerate(mp3_paths):
        if p is np.NaN:
            x = np.zeros((1, 1, num_rows, num_cols), dtype=float)
        else:
            print(p)
            path = "./data/mp3/%s" % (p)
            x = ap.compute_melgram(path)
        examples = np.append(examples, x, axis=0)



    """
    examples = np.empty((0, num_rows, num_cols), dtype=float)
    for i, p in enumerate(mp3_paths):
        if p is np.NaN:
            x = np.zeros((num_rows, num_cols), dtype=float)
        else:
            print(p)
            path="./data/mp3/%s.mels.tsv" %(p)
            df2 = pd.read_csv(path, delimiter="\t")
            x = df2.as_matrix()

        x = x[np.newaxis, :]
        examples = np.append(examples, x, axis=0)
    """
    return examples


def load_mfcc_examples(fname="./data/clip_info_final.csv", limit=10, num_rows=127, num_cols=1255):
    df = pd.read_csv(fname, delimiter="\t")
    print(df.info())
    mp3_paths = df["mp3_path"][:limit]

    examples = np.empty((0, num_rows, num_cols), dtype=float)
    for i, p in enumerate(mp3_paths):
        if p is np.NaN:
            x = np.zeros((num_rows, num_cols), dtype=float)
        else:
            print(p)
            path="./data/mp3/%s.mfcc.tsv" %(p)
            df2 = pd.read_csv(path, delimiter="\t")
            x = df2.as_matrix()

        x = x[np.newaxis, :]
        examples = np.append(examples, x, axis=0)

    return examples


if __name__ == "__main__":
    examples = data_loader.load_mel_examples(1000)
    labels = data_loader.load_labeles(1000)

    print(labels.shape)
    print(examples.shape)
