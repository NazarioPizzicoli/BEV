import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset

sys.path.append(str(Path.cwd().parent))

from src.utils.text_utils import decode_question, encode_question, load_glove


class BEVQADataset(Dataset):
    """
    Dataset:
        - BEV Features
        - Question
        - Answer (tokenized)
    """

    def __init__(self, bev_dir, json_path, glove):
        self.bev_dir = Path(bev_dir)

        with open(json_path, "r") as f:
            data = json.load(f)

        # Lista allineata: (sample_token, question, answer)
        self.samples = [
            (q["sample_token"], q["question"], q["answer"])
            for q in data["questions"]
            if (self.bev_dir / f"{q['sample_token']}.pt").exists()
        ]

        self.glove = load_glove(glove)
        self.answer2idx = decode_question(json_path)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        token, question, answer = self.samples[idx]

        features = torch.load(self.bev_dir / f"{token}.pt", map_location="cpu")
        features = (
            features["features_fused"].squeeze(0).to(torch.float32)
        )  # [1,128,200,200] → [128,200,200]

        question_enc = encode_question(question, self.glove)  # [35,300]

        answer_idx = self.answer2idx.get(answer, -1)
        return features, question_enc, torch.tensor(answer_idx, dtype=torch.long)
