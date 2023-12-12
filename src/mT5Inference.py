import argparse
import json
import os

import pandas as pd
import torch
from transformers import AutoTokenizer, MT5ForConditionalGeneration

MT5_SMALL = "google/mt5-small"
MT5_LARGE = "google/mt5-large"

DIR = "/data/tir/projects/tir5/users/shailyjb/anlp-figlang/mt5_large_textoptions_seq2seq/checkpoint-3660"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=DIR,
    )
    parser.add_argument("--tokenizer", type=str, default=MT5_LARGE)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DIR + "/test_out",
    )
    parser.add_argument(
        "--data_dir", type=str, default="/home/shailyjb/anlp-figlang/data"
    )
    # parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--split", type=str, default="test")
    return parser.parse_args()


def get_model(model_name):
    return MT5ForConditionalGeneration.from_pretrained(model_name)


def _prepare_inference_example(tokenizer, example):
    # Create input by concatenating question and options
    metaphor = example["startphrase"]
    meaning_0 = example["ending1"]
    meaning_1 = example["ending2"]

    # Ensuring they all have full stops for uniformity.
    if metaphor[-1] != ".":
        metaphor += "."
    if meaning_0[-1] != ".":
        meaning_0 += "."
    if meaning_1[-1] != ".":
        meaning_1 += "."

    # Create prompt that will be input to the model
    input_text = f"select the correct meaning for the metaphor:\n\nmetaphor: '{metaphor}'\noption a)'{meaning_0}'\option b) '{meaning_1}'\ncorrect option: <extra_id_0>"
    # label = "option a" if example["labels"] == 0 else "option b"
    # label = "a" if example["labels"] == 0 else "b"
    if example["labels"] == 0:
        label = "option a"
    elif example["labels"] == 1:
        label = "option b"
    # label = str(example["labels"])
    # print(label)
    input_label = "<extra_id_0> {label}".format(label=label)

    # input_text = f"select the best meaning of the metaphor: '{metaphor}' from the meanings: 0) '{meaning_0}' or 1) '{meaning_1}'."

    # Tokenize input text
    tokenized_input = tokenizer(
        input_text,
        return_tensors="pt",
        # padding="max_length",
        # truncation=True,
        # max_length=128,
    )
    tokenized_labels = tokenizer(
        # str(example["labels"]),
        input_label,
        return_tensors="pt",
        # padding="max_length",
        # truncation=True,
        # max_length=128,
    )
    # print(type(tokenized_input.input_ids))
    # print(tokenized_input)
    # print(tokenized_labels)
    return {
        # "input_ids": tokenized_input.input_ids.flatten(),
        "input_ids": tokenized_input.input_ids,
        # "labels": tokenized_labels.input_ids.flatten(),
        "labels": tokenized_labels.input_ids.flatten(),
    }


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.ckpt_path, local_files_only=True
    )
    for lang in "hi,id,jv,kn,su,sw,yo".split(","):
        # for lang in "en".split(","):
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
            example = {
                "startphrase": row["startphrase"],
                "ending1": row["ending1"],
                "ending2": row["ending2"],
                "labels": row["labels"],
            }

            tokenized_example = _prepare_inference_example(tokenizer, example)

            output = model.generate(tokenized_example["input_ids"])
            output_text = tokenizer.decode(output[0], skip_special_tokens=True)
            decoded_label = tokenizer.decode(
                tokenized_example["labels"], skip_special_tokens=True
            )

            # print(example)
            # print(tokenized_example)
            # print("tokenized input label: ", tokenized_example["labels"])
            # print("decoded tokenized input label: ", decoded_label)
            # print("model output: ", output)
            # print("decoded output text: ", output_text)
            # print(
            #     "decoded output text == decoded label: ", output_text == decoded_label
            # )
            if output_text == decoded_label:
                metrics["correct"].append(
                    (
                        example["startphrase"],
                        example["ending1"],
                        example["ending2"],
                        decoded_label,
                        output_text,
                    )
                )
            else:
                metrics["wrong"].append(
                    (
                        example["startphrase"],
                        example["ending1"],
                        example["ending2"],
                        decoded_label,
                        output_text,
                    )
                )
        assert len(metrics["correct"]) + len(metrics["wrong"]) == len(df)
        metrics["accuracy"] = len(metrics["correct"]) / len(df)
        json.dump(metrics, open(output_path, "w"), indent=4)
