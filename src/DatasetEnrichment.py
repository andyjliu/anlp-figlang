import os

import openai
from retry import retry
import pandas as pd
from tqdm import tqdm

from src.Translator import Translator

openai.api_key = open(os.path.join(os.path.expanduser("~"), ".openai_key")).read().strip()


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
    return openai.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": input_content
            }
        ]
    )

def get_gpt_answer(gpt_response) -> str:
    return gpt_response.choices[0].message.content


def build_enhanced_startphrase(row, characteristics_suffix_column_name='characteristics_suffix'):
    startphrase = row['startphrase']

    characteristics_suffix = row[characteristics_suffix_column_name]

    enhanced_startphrase = f"{startphrase}. {characteristics_suffix}"
    enhanced_startphrase = enhanced_startphrase.replace("..", ".")

    return enhanced_startphrase

def build_characteristics_suffix(row):
    vehicle = row['vehicle']
    characteristics = row['vehicle_characteristics']

    return f"Characteristics of {vehicle} include: {characteristics}."

def enrich_df(df_path, enhanced_df_path, original_startphrase_column_name='startphrase', is_test=False, translator=None, force_rebuild=False):
    if not os.path.exists(enhanced_df_path) or force_rebuild:
        df = pd.read_csv(df_path)

        df['vehicle'] = df[original_startphrase_column_name].apply(lambda x: vehicle_detector.get_vehicle(x))

        print("finished getting vehicles")

        df['vehicle_characteristics'] = df['vehicle'].apply(
            lambda x: figurative_characteristics_expander.get_k_characteristics(x))

        print("finished getting vehicle characteristics")

        df['characteristics_suffix'] = df.apply(lambda x: build_characteristics_suffix(x), axis=1)

        if is_test:
            characteristics_suffixes = df['characteristics_suffix'].to_list()
            df['characteristics_suffix_final'] = translator.get_translations(characteristics_suffixes, source_lang='en-US', target_lang=lang)
            print("finished getting characteristics_suffix_final")

        characteristics_suffix_column_name = 'characteristics_suffix' if not is_test else 'characteristics_suffix_final'

        df['startphrase_enhanced'] = df.apply(lambda x: build_enhanced_startphrase(x, characteristics_suffix_column_name), axis=1)

        df.to_csv(enhanced_df_path)


if __name__ == "__main__":
    translator = Translator()
    vehicle_detector = VehicleDetector()
    figurative_characteristics_expander = FigurativeCharacteristicsExpander()

    test_langs = ['id', 'jv', 'kn', 'su', 'sw', 'yo']
    for lang in test_langs:
        print(f"Starting {lang}...")
        df_path = f"../data/test/{lang}.csv"
        enhanced_df_path = f"../data/test_enhanced/{lang}.csv"

        test_df = pd.read_csv(df_path)

        startphrases = test_df['startphrase'].to_list()
        test_df['en_startphrase'] = translator.get_translations(startphrases, source_lang=lang, target_lang='en-US')

        test_df.to_csv(enhanced_df_path)

        enrich_df(enhanced_df_path, enhanced_df_path, original_startphrase_column_name='en_startphrase', is_test=True,
                  translator=translator, force_rebuild=True)

    for split in ['train', 'validation']:
        df_path = f"../data/{split}/en.csv"
        enhanced_df_path = f'../data/{split}_enhanced/en.csv'

        enrich_df(df_path, enhanced_df_path)