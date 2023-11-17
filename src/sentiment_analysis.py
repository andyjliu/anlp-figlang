import argparse
import json
import random

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
        print(lang)
        input_path = f"{args.data_dir}/{lang}.json"
        data = json.load(open(input_path, "r"))
        results = {
            "correct": {
                "true_same_sentiment": 0,
                "true_diff_sentiment": 0,
                "model_same_sentiment": 0,
                "model_diff_sentiment": 0,
                "true_same_sentiment_examples": [],
                "true_diff_sentiment_examples": [],
                "model_same_sentiment_examples": [],
                "model_diff_sentiment_examples": [],
            },
            "wrong": {
                "true_same_sentiment": 0,
                "true_diff_sentiment": 0,
                "model_same_sentiment": 0,
                "model_diff_sentiment": 0,
                "true_same_sentiment_examples": [],
                "true_diff_sentiment_examples": [],
                "model_same_sentiment_examples": [],
                "model_diff_sentiment_examples": [],
            },
        }
        for key in ["correct", "wrong"]:
            for datapoint in data[key]:
                startphrase = datapoint[0]
                predicted_class = datapoint[-1]
                true_class = datapoint[-2]
                if predicted_class == 0:
                    model_prediction = datapoint[1]
                else:
                    model_prediction = datapoint[2]
                if true_class == 0:
                    true_prediction = datapoint[1]
                else:
                    true_prediction = datapoint[2]
                startphrase_sentiment = get_sentiment(startphrase)
                model_prediction_sentiment = get_sentiment(model_prediction)
                true_prediction_sentiment = get_sentiment(true_prediction)
                if startphrase_sentiment == model_prediction_sentiment:
                    results[key]["model_same_sentiment"] += 1
                    results[key]["model_same_sentiment_examples"].append(datapoint)
                else:
                    results[key]["model_diff_sentiment"] += 1
                    results[key]["model_diff_sentiment_examples"].append(datapoint)
                if startphrase_sentiment == true_prediction_sentiment:
                    results[key]["true_same_sentiment"] += 1
                    results[key]["true_same_sentiment_examples"].append(datapoint)
                else:
                    results[key]["true_diff_sentiment"] += 1
                    results[key]["true_diff_sentiment_examples"].append(datapoint)
            for k in [
                "model_same_sentiment_examples",
                "model_diff_sentiment_examples",
                "true_same_sentiment_examples",
                "true_diff_sentiment_examples",
            ]:
                results[key][k] = random.sample(
                    results[key][k], min(5, len(results[key][k]))
                )
        output_path = f"{args.data_dir}/{lang}_sentiment.json"
        json.dump(results, open(output_path, "w"), indent=4)
