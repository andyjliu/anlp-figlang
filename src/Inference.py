import argparse
import json
import os

import evaluate
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

    return parser.parse_args()


def get_model(ckpt_path):
    return AutoModelForMultipleChoice.from_pretrained(ckpt_path)


# def inference(data, output_path, model):
#     for datapoint in data:
#         print(datapoint)
#         datapoint = json.loads(datapoint)
#         print(type(datapoint))
#         outputs = model(
#             input_ids=datapoint["input_ids"],
#             attention_mask=datapoint["attention_mask"],
#             labels=datapoint["labels"],
#         )
#         prediction = outputs.logits.argmax(dim=-1)
#         print(prediction)


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = AutoModelForMultipleChoice.from_pretrained(
        args.ckpt_path, local_files_only=True
    )
    for lang in "hi,id,jv,kn,su,sw,yo".split(","):
        path = f"{args.data_dir}/test/{lang}.csv"
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
            predicted_class = logits.argmax().item()
            if true_label == predicted_class:
                metrics["correct"].append((startphrase, ending1, ending2, true_label))
            else:
                metrics["wrong"].append((startphrase, ending1, ending2, true_label))
        assert len(metrics["correct"]) + len(metrics["wrong"]) == len(df)
        metrics["accuracy"] = len(metrics["correct"]) / len(df)
        json.dump(metrics, open(output_path, "w"), indent=4)

    # mabl_dataset_dict = MablDatasetDict(data_dir=args.data_dir, splits="test")
    # raw_datasets = mabl_dataset_dict.get_dataset_dict()
    # mabl_dataset_dict.tokenize_dataset(tokenizer, drop_columns=False)
    # processed_datasets = mabl_dataset_dict.get_dataset_dict()
    # data_collator = DataCollatorForMultipleChoice(tokenizer, padding=True)
    # for lang in processed_datasets:
    #     print(lang)
    #     test_dataset = processed_datasets[lang]
    #     test_dataloader = DataLoader(
    #         test_dataset,
    #         collate_fn=data_collator,
    #         batch_size=32,
    #     )
    #     raw_test_dataloader = DataLoader(raw_datasets[lang], batch_size=32)
    #     output_path = f"{args.output_dir}/{lang}.json"
    #     if os.path.exists(output_path):
    #         open(output_path, "w").close()

    #     for step, (batch, raw_batch) in enumerate(
    #         zip(test_dataloader, raw_test_dataloader)
    #     ):
    #         with torch.no_grad():
    #             outputs = model(
    #                 input_ids=batch["input_ids"],
    #                 attention_mask=batch["attention_mask"],
    #                 labels=batch["labels"],
    #             )
    #         predictions = outputs.logits.argmax(dim=-1)
    #         startphrases = raw_batch["startphrase"]
    #         ending1s = raw_batch["ending1"]
    #         ending2s = raw_batch["ending2"]
    #         with open(output_path, "a") as f:
    #             for startphrase, ending1, ending2, label, prediction in zip(
    #                 startphrases, ending1s, ending2s, batch["labels"], predictions
    #             ):
    #                 f.write(f"{startphrase},{ending1},{ending2},{label},{prediction}\n")
