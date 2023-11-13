import os
from itertools import chain

from datasets import Dataset, DatasetDict, load_dataset


class MablDatasetDict(DatasetDict):
    def __init__(self, data_dir="data"):
        super().__init__()

        self.dataset_dict = {}
        for split in ["train", "validation"]:
            file_path = f"{data_dir}/{split}/en.csv"
            self.dataset_dict[split] = Dataset.from_csv(file_path)
            # self.dataset_dict[split] = load_dataset("csv", file_path)

        test_dir = f"{data_dir}/test"
        for file_name in os.listdir(test_dir):
            lang_short_name = file_name.split("_")[0]
            file_path = f"{test_dir}/{file_name}"
            self.dataset_dict[f"test-{lang_short_name}"] = Dataset.from_csv(file_path)
            # self.dataset_dict[f"test-{lang_short_name}"] = load_dataset(
            #     "csv", file_path
            # )

    def get_dataset_dict(self):
        return self.dataset_dict

    def tokenize_dataset(self, tokenizer):
        def tokenize_examples(examples):
            fig_phrases = [
                [fig_phrase] * 2 for fig_phrase in examples["startphrase"]
            ]

            answers = [
                [endings_tuple[0], endings_tuple[1]] for endings_tuple in zip(examples["ending1"], examples["ending2"])
            ]

            flattened_fig_phrases = sum(fig_phrases, [])
            flattened_answers = sum(answers, [])

            # Tokenize
            batch_encoding = tokenizer(
                flattened_fig_phrases,
                flattened_answers,
                max_length=128,
                padding=False,
                truncation=True
            )   # currently max_len and padding and truncation are hardcoded, but we can flagify them later.

            unflattened_tokenized_batch_encoding = {}
            for batch_encoding_key, values in batch_encoding.items():
                values_for_instance = [values[i: i + 2] for i in range(0, len(values), 2)]
                unflattened_tokenized_batch_encoding[batch_encoding_key] = values_for_instance

            labels = examples["labels"]
            unflattened_tokenized_batch_encoding["labels"] = labels

            return unflattened_tokenized_batch_encoding

        for key in ["train", "validation"]:
            dataset = self.dataset_dict[key]
            tokenized_dataset = dataset.map(tokenize_examples, batched=True)
            self.dataset_dict[key] = tokenized_dataset


if __name__ == "__main__":
    print(MablDatasetDict(data_dir="../data").get_dataset_dict())
