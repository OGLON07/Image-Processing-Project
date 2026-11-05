"""Kaggle Submission entrypoint.
Calls src.inference.generate_submission to output submission.csv.
"""

from src.inference import generate_submission
import os
import torch

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = "artifacts/model_fold_0.pt" if os.path.exists("artifacts/model_fold_0.pt") else None
    generate_submission(model_checkpoint=ckpt, device=device)

