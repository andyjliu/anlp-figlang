import json
import pandas as pd

def get_json(fp):
    with open(fp, 'r') as f:
        j = json.load(f)
    return(j)

def convert_to_df_rows(json_obj, lang, label):
    prompt, opt1, opt2, pred, act, langs, labels = [], [], [], [], [], [], []
    for j in json_obj:
        prompt.append(j[0])
        opt1.append(j[1])
        opt2.append(j[2])
        pred.append(j[3])
        act.append(j[4])
        langs.append(lang)
        labels.append(label)
    return(pd.DataFrame.from_dict({'prompt':prompt, 'opt1':opt1, 'opt2':opt2, 'pred':pred, 'act':act, 'lang':langs, 'label':labels}))

langs = ['hi', 'id', 'jv', 'kn', 'su', 'sw', 'yo']
labels = ['correct', 'wrong']

df = pd.DataFrame(columns=['prompt', 'opt1', 'opt2', 'pred', 'act', 'lang', 'label'])
for lang in langs:
    lang_json = get_json(f'xlmr_large_end_ckpt/test_out/{lang}.json')
    for label in labels:
        new_df = convert_to_df_rows(lang_json[label], lang, label)
        df = pd.concat([df, new_df])

df.to_csv('all_untranslated.csv')
