#!/usr/bin/env python3

import argparse
import numpy as np

import eer

#parser = argparse.ArgumentParser()
#parser.add_argument("speaker_embeddings")

def cos_score(x: np.ndarray, y: np.ndarray):
    """Compute the cosine score between matrices x and y efficiently,
    where x is shape (N_train, N_embed) and y is shape (N_test, N_embed)"""

    assert x.shape[1] == y.shape[1], "Embedding dimension must match"
    xn = np.sqrt((x * x).sum(axis=1, keepdims=True))
    yn = np.sqrt((y * y).sum(axis=1, keepdims=True))
    return np.dot(x, y.transpose()) / xn / yn.transpose()

def compeer(embeddings,speakers):
 #   args = parser.parse_args()
 #   embeddings = list()
 #   speakers = list()
 #   for line in open(args.speaker_embeddings):
 #       speaker_id, embedding = line.strip().split(None, 1)
 #       embeddings.append(eval(embedding))
 #       speakers.append(speaker_id)
    embeddings = np.array(embeddings)
    speakers = np.array(speakers)
#    print("Embeddings shape", embeddings.shape)
    scores = cos_score(embeddings, embeddings)
    labels = np.expand_dims(speakers, 1) == speakers
    assert scores.shape == labels.shape
#    print("Scores shape", scores.shape)
    upper_tri = np.triu(np.ones(scores.shape, dtype=bool), k=1)
    flat_scores = scores[upper_tri]
    flat_labels = labels[upper_tri]
#    print("Flat scores shape", flat_scores.shape)
#    print(flat_scores)
#    print(flat_labels)
    print(f"Equal Error Rate = {eer.eer(flat_scores, flat_labels):4.1%}")
    return eer.eer(flat_scores, flat_labels)

if __name__ == "__main__":
    eer()
