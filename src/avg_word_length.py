import argparse
import json

import pandas as pd
from transformers import AutoTokenizer

HF_MODEL_NAME_XLMR_LARGE = "xlm-roberta-large"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tokenizer", type=str, default=HF_MODEL_NAME_XLMR_LARGE)
    parser.add_argument("--data_dir", type=str, default="../data")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for lang in "hi,id,jv,kn,su,sw,yo".split(","):
        path = f"{args.data_dir}/{lang}.json"
        data = json.load(open(path, "r"))
        avg_word_lengths = []
        for key in ["correct", "wrong"]:
            for data_point in data[key]:
                startphrase = data_point[0]
                num_tokens = len(tokenizer.tokenize(startphrase))
                num_words = len(startphrase.split(" "))
                avg_word_length = num_tokens / num_words
                avg_word_lengths.append(
                    (key, startphrase, num_words, num_tokens, avg_word_length)
                )
        output_path = f"{args.data_dir}/{lang}_avg_word_lengths.csv"
        pd.DataFrame(
            avg_word_lengths,
            columns=[
                "key",
                "startphrase",
                "num_words",
                "num_tokens",
                "avg_word_length",
            ],
        ).to_csv(output_path, index=False)
