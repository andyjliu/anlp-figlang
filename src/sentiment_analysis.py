import argparse
import json
import random

import pandas as pd
from transformers import pipeline

from src import statstools

sentiment_pipeline = pipeline("sentiment-analysis")


def get_sentiment(text):
    return sentiment_pipeline(text)[0]["label"]


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="../xlmr_large_end_ckpt/translate_test_out")

    return parser.parse_args()


if __name__ == "__main__":
    langs = "hi,id,jv,kn,su,sw,yo".split(",")
    args = parse_args()
    for lang in langs:
        sentiment_dict = {}

        sentiment_dict[f'{lang}_prediction_correct'] = []
        sentiment_dict[f'{lang}_predicted_sentiments_match'] = []

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
                sentiment_dict[f'{lang}_prediction_correct'].append(1 if key == 'correct' else 0)
                if startphrase_sentiment == true_prediction_sentiment:
                    results[key]["true_same_sentiment"] += 1
                    results[key]["true_same_sentiment_examples"].append(datapoint)
                    sentiment_dict[f'{lang}_predicted_sentiments_match'].append(1)
                else:
                    results[key]["true_diff_sentiment"] += 1
                    results[key]["true_diff_sentiment_examples"].append(datapoint)
                    sentiment_dict[f'{lang}_predicted_sentiments_match'].append(0)
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

        # t tests
        sentiment_df = pd.DataFrame.from_dict(sentiment_dict)

        res = {
            'avg_predicted_sentiments_match_when_prediction_correct': [],
            'avg_predicted_sentiments_match_when_prediction_incorrect': [],
            'direction_of_effect': [],
            'effect_size': [],
            'p_value': []
        }

        t_test_results = statstools.ttestSummary(sentiment_df, f"{lang}_prediction_correct", f"{lang}_predicted_sentiments_match")
        res['avg_predicted_sentiments_match_when_prediction_incorrect'].append(round(t_test_results['mean_0'], 3))
        res['avg_predicted_sentiments_match_when_prediction_correct'].append(round(t_test_results['mean_1'], 3))
        res['direction_of_effect'].append('prediction_correct' if t_test_results['d'] < 0 else 'prediction_incorrect')
        res['effect_size'].append(round(abs(t_test_results['d']), 3))
        res['p_value'].append(round(t_test_results['p'], 3))

        print(res)