import os
from itertools import chain

from datasets import Dataset, DatasetDict, load_dataset


class MablDatasetDict(DatasetDict):
    def __init__(self, data_dir="data"):
        super().__init__()

        self.dataset_dict = {}
        for split in ["train", "validation"]:
            file_path = f"{data_dir}/{split}/en.csv"
            # self.dataset_dict[split] = Dataset.from_csv(file_path)
            self.dataset_dict[split] = load_dataset("csv", file_path)

        test_dir = f"{data_dir}/test"
        for file_name in os.listdir(test_dir):
            lang_short_name = file_name.split("_")[0]
            file_path = f"{test_dir}/{file_name}"
            # self.dataset_dict[f"test-{lang_short_name}"] = Dataset.from_csv(file_path)
            self.dataset_dict[f"test-{lang_short_name}"] = load_dataset(
                "csv", file_path
            )

    def get_dataset_dict(self):
        return self.dataset_dict

    def tokenize(self, tokenizer):
        def tokenize_examples(examples):
            first_sentences = [
                [startphrase] * 2 for startphrase in examples["startphrase"]
            ]
            second_sentences = [
                [examples[end][i] for end in ["ending1", "ending2"]]
                for i in range(len(examples["startphrase"]))
            ]
            labels = examples["labels"]
            first_sentences = sum(first_sentences, [])
            second_sentences = sum(second_sentences, [])

            tokenized_examples = tokenizer(
                first_sentences,
                second_sentences,
                max_length=128,
                padding=False,
                truncation=True,
            )  # currently max_len and padding and truncation are hardcoded, but we can flagify them later.

            tokenized_inputs = {
                k: [v[i : i + 2] for i in range(0, len(v), 2)]
                for k, v in tokenized_examples.items()
            }
            tokenized_inputs["labels"] = labels
            return tokenized_inputs

        for key in ["train", "validation"]:
            dataset = self.dataset_dict[key]
            tokenized_dataset = dataset.map(tokenize_examples, batched=True)
            self.dataset_dict[key] = tokenized_dataset


if __name__ == "__main__":
    print(MablDatasetDict(data_dir="../data").get_dataset_dict())
