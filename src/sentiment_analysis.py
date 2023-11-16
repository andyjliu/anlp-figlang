import argparse
import json

from transformers import pipeline

sentiment_pipeline = pipeline("sentiment-analysis")


def get_sentiment(text):
    return sentiment_pipeline(text)[0]["label"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../data")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    for lang in "hi,id,jv,kn,su,sw,yo".split(","):
        input_path = f"{args.data_dir}/{lang}.json"
        data = json.load(open(input_path, "r"))
        results = {
            "correct": {"num_same_sentiment": 0, "num_diff_sentiment": 0},
            "wrong": {"num_same_sentiment": 0, "num_diff_sentiment": 0},
        }
        for key in ["correct", "wrong"]:
            for datapoint in data[key]:
                startphrase = datapoint[0]
                predicted_class = datapoint[-1]
                if predicted_class == 0:
                    prediction = datapoint[1]
                else:
                    prediction = datapoint[2]
                startphrase_sentiment = get_sentiment(startphrase)
                prediction_sentiment = get_sentiment(prediction)
                if startphrase_sentiment == prediction_sentiment:
                    results[key]["num_same_sentiment"] += 1
                else:
                    results[key]["num_diff_sentiment"] += 1
        output_path = f"{args.data_dir}/{lang}_sentiment.json"
        json.dump(results, open(output_path, "w"))
