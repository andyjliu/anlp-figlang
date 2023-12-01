import os

import openai
from retry import retry
import pandas as pd

openai.api_key = open(os.path.join(os.path.expanduser("~"),".openai_key")).read().strip()


class VehicleDetector:

    def __init__(self):
        pass

    def get_vehicle(self, metaphor):
        prompt = (f"You will be presented a short sentence containing figurative language, such as a metaphor or "
                  f"simile. In each case, one noun (the tenor) is described by a different noun phrase (the vehicle). "
                  f"Typically, the tenor is the first noun in the phrase and the vehicle is the last noun in the "
                  f"phrase. However, sometimes there are multiple nouns in a phrase acting as the vehicle. In this "
                  f"case, usually some of the nouns in the vehicle phrase describe characteristics of a single core "
                  f"noun in the vehicle phrase. Your task is to output the single noun taken directly from the phrase "
                  f"that best represents the vehicle. Never output an adjective unless it is part of a proper noun, "
                  f"and don't preface your answer with 'vehicle:' or anything similar. \n\n{metaphor}")

        gpt_response = get_gpt_response(input_content=prompt)

        gpt_answer = get_gpt_answer(gpt_response)

        return gpt_answer


class FigurativeCharacteristicsExpander:

    def __init__(self):
        pass

    def get_k_characteristics(self, word, k=3):
        prompt = (f"Consider the characteristics of the word '{word}'? Your output should be a comma-separated list of "
                  f"adjectives of length {k}, where each adjective represents a characteristic of the original word. "
                  f"In general, you should favor figurative characteristics over literal characteristics. Don't "
                  f"include any extra preface like 'characteristics: ' or brackets in your output.")

        gpt_response = get_gpt_response(input_content=prompt)

        gpt_answer = get_gpt_answer(gpt_response)

        return gpt_answer


@retry(tries=5, jitter=1, backoff=2)
def get_gpt_response(input_content: str, model="gpt-3.5-turbo-0613"):
    return openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input_content
            }
        ]
    )


def get_gpt_answer(gpt_response) -> str:
    return gpt_response['choices'][0]['message']['content']


def build_enhanced_startphrase(row):
    startphrase = row['startphrase']
    vehicle = row['vehicle']
    characteristics = row['vehicle_characteristics']

    enhanced_startphrase = f"{startphrase}. Characteristics of {vehicle} include: {characteristics}."
    enhanced_startphrase = enhanced_startphrase.replace("..", ".")

    return enhanced_startphrase


if __name__ == "__main__":
    vehicle_detector = VehicleDetector()

    figurative_characteristics_expander = FigurativeCharacteristicsExpander()

    if not os.path.exists('../data/train_enhanced/en.csv'):
        en_train_df = pd.read_csv("../data/train/en.csv")

        en_train_df['vehicle'] = en_train_df['startphrase'].apply(lambda x: vehicle_detector.get_vehicle(x))

        en_train_df['vehicle_characteristics'] = en_train_df['vehicle'].apply(lambda x: figurative_characteristics_expander.get_k_characteristics(x))

        en_train_df['startphrase_enhanced'] = en_train_df.apply(lambda x: build_enhanced_startphrase(x), axis=1)

        en_train_df.to_csv("../data/train_enhanced/en.csv")