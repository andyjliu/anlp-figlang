import pandas as pd
from transformers import AutoTokenizer

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    langs = ['hi', 'id', 'jv', 'kn', 'su', 'sw', 'yo']

    lang_dfs = {}
    object_dicts = {}
    token_dict = {}

    for lang in langs:
        path = f"../data/syntax_chunked/syntax_tagged_{lang}.csv"
        df = pd.read_csv(path, on_bad_lines='skip')

        df['tokens'] = df['z'].apply(lambda x: tokenizer.tokenize(str(x)))
        df['num_words'] = df['z'].apply(lambda x: len(str(x).split(" ")))
        df['num_tokens'] = df['tokens'].apply(lambda x: len(x))
        df['avg_word_length'] = df.apply(lambda x: x['num_tokens'] / x['num_words'], axis=1)

        tok_set = set()
        object_dict = {}
        for i, row in df.iterrows():
            for tok in row['tokens']:
                if tok not in object_dict:
                    object_dict[tok] = 0
                object_dict[tok] += 1

            tok_set.update(row['tokens'])

        object_dicts[lang] = object_dict

        for tok in tok_set:
            if tok not in token_dict:
                token_dict[tok] = 0
            token_dict[tok] += 1

        lang_dfs[lang] = df

    print(object_dicts)
    print(token_dict)

    # get object commonness score for each language
    for lang in langs:
        commonness_score = 0
        for key in object_dicts[lang].keys():
            commonness_score += token_dict[key]

        normalized_commonness_score = commonness_score / len(object_dicts[lang].keys())

        print(f"{lang} translated object commonness: {round(normalized_commonness_score, 2)}")

    print("=================")
    for lang in langs:
        avg_word_lengths = sum(lang_dfs[lang]['avg_word_length'])
        score = avg_word_lengths / len(lang_dfs[lang].index)
        print(f"{lang} translated avg word length (objects only): {round(score, 2)}")

    print("=============")
    # TODO - compare overlap between each language's translated train dataset and the English test set

