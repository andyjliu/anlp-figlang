import pandas as pd

if __name__ == "__main__":
    """
    This script generates train dataset files for the experiments involving "few-shot" supplements of non-English 
    instances. Each generated dataset has a corresponding test file that removes the 'borrowed' test set examples from 
    the test set. The git repo for the original paper does not include all 'merged' train datasets (defined as the 
    original english training dataset plus the k examples from the non-English language) presented in the paper. It 
    does, however, include the supplementary non-English examples and corresponding test set dataset files, so we pull 
    those into the repo manually, then use the supplementary files along with the original English training dataset to 
    generate the missing 'merged' training files for the few-shot experiments.
    """
    en_train_df = pd.read_csv('../data/train/en.csv')

    for lang in ['hi', 'id', 'jv', 'kn', 'su', 'sw', 'yo']:
        for k in [2, 10, 20, 30, 40, 50]:
            en_train_copy_df = en_train_df.copy(deep=True)

            lang_train_df = pd.read_csv(f'../data/few_shot/train_supplements/{lang}/{lang}_{k}.csv')

            merged_train_df = pd.concat([en_train_copy_df, lang_train_df], ignore_index=True)

            merged_train_df.to_csv(f'../data/few_shot/train_merged/{lang}/{lang}_{k}.csv')