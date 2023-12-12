import argparse
import json
import os

import pandas as pd
import torch
from DataCollator import DataCollatorForMultipleChoice
from MablDatasetDict import MablDatasetDict
from torch.utils.data import DataLoader
from transformers import AutoModelForMultipleChoice, AutoTokenizer

HF_MODEL_NAME_XLMR_LARGE = "xlm-roberta-large"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt_path", type=str, default="")
    parser.add_argument("--tokenizer", type=str, default=HF_MODEL_NAME_XLMR_LARGE)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--sent', action='store-true')
    parser.add_argument('--threshold', type=float, default=0.6)

    return parser.parse_args()


def get_model(ckpt_path):
    return AutoModelForMultipleChoice.from_pretrained(ckpt_path)


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.ckpt_path, local_files_only=True
    )

    if 'validation' in args.split:
        langs = ['en']
    else:
        langs = "hi,id,jv,kn,su,sw,yo".split(",")

    for lang in langs:
        path = f"{args.data_dir}/{args.split}/{lang}.csv"
        df = pd.read_csv(path, sep=",", header=0)
        output_path = f"{args.output_dir}/{lang}.json"
        if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)
        metrics = {
            "correct": [],
            "wrong": [],
        }
        for index, row in df.iterrows():
            startphrase = row["startphrase"]
            ending1 = row["ending1"]
            ending2 = row["ending2"]
            true_label = row["labels"]
            inputs = tokenizer(
                [[startphrase, ending1], [startphrase, ending2]],
                return_tensors="pt",
                padding=True,
            )
            labels = torch.tensor(0).unsqueeze(0)
            outputs = model(
                **{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels
            )
            logits = outputs.logits
            if args.sent:
                confidence = torch.softmax(logits).max().item()
                if confidence < args.threshold:
                    predicted_class = row['sent_label']
                else:
                    predicted_class = logits.argmax().item()
                    
            else:
                predicted_class = logits.argmax().item()
            if true_label == predicted_class:
                metrics["correct"].append(
                    (startphrase, ending1, ending2, true_label, predicted_class)
                )
            else:
                metrics["wrong"].append(
                    (startphrase, ending1, ending2, true_label, predicted_class)
                )
        assert len(metrics["correct"]) + len(metrics["wrong"]) == len(df)
        metrics["accuracy"] = len(metrics["correct"]) / len(df)
        json.dump(metrics, open(output_path, "w"), indent=4)
