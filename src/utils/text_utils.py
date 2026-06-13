# text_utils.py
import numpy as np
import torch
import json


def load_glove(glove_path):
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.array(values[1:], dtype=np.float32)
            glove[word] = vector
    return glove


def encode_question(question, glove, max_len=30):
    tokens = question.lower().split()
    vectors = []
    for token in tokens:
        vec = glove.get(token, np.zeros(300, dtype=np.float32))
        vectors.append(vec)
    # padding
    while len(vectors) < max_len:
        vectors.append(np.zeros(300, dtype=np.float32))
    vectors = vectors[:max_len]  # truncate se > max_len
    return torch.tensor(np.array(vectors), dtype=torch.float32)  # [30, 300]


def decode_question(json_path):
    with open(json_path) as f:
        data = json.load(f)
    unique_answers = sorted(set(q["answer"] for q in data["questions"]))
    answer2idx = {ans: idx for idx, ans in enumerate(unique_answers)}
    idx2answer = {idx: ans for ans, idx in answer2idx.items()}
    return answer2idx