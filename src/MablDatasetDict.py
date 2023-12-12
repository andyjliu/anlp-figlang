from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


class MablDatasetDict(DatasetDict):
    def __init__(self, train_file, validation_file):
        super().__init__()
        self.dataset_dict = {}

        self.dataset_dict["train"] = Dataset.from_csv(train_file)
        self.dataset_dict["validation"] = Dataset.from_csv(validation_file)

    def get_dataset_dict(self):
        return self.dataset_dict

    def tokenize_dataset(self, tokenizer, drop_columns=True):
        def tokenize_examples(examples):
            fig_phrases = [[fig_phrase] * 2 for fig_phrase in examples["startphrase"]]

            answers = [
                [endings_tuple[0], endings_tuple[1]]
                for endings_tuple in zip(examples["ending1"], examples["ending2"])
            ]

            flattened_fig_phrases = sum(fig_phrases, [])
            flattened_answers = sum(answers, [])

            # Tokenize
            batch_encoding = tokenizer(
                flattened_fig_phrases,
                flattened_answers,
                max_length=128,
                padding=False,
                truncation=True,
            )  # currently max_len and padding and truncation are hardcoded, but we can flagify them later.

            unflattened_tokenized_batch_encoding = {}
            for batch_encoding_key, values in batch_encoding.items():
                values_for_instance = [
                    values[i : i + 2] for i in range(0, len(values), 2)
                ]
                unflattened_tokenized_batch_encoding[
                    batch_encoding_key
                ] = values_for_instance

            labels = examples["labels"]
            unflattened_tokenized_batch_encoding["labels"] = labels

            return unflattened_tokenized_batch_encoding

        for key in self.dataset_dict:
            dataset = self.dataset_dict[key]
            if drop_columns:
                tokenized_dataset = dataset.map(
                    tokenize_examples, batched=True, remove_columns=dataset.column_names
                )
            else:
                tokenized_dataset = dataset.map(tokenize_examples, batched=True)
            self.dataset_dict[key] = tokenized_dataset


if __name__ == "__main__":
    print(
        MablDatasetDict(
            train_file="/home/shailyjb/anlp-figlang/data/train/en.csv",
            validation_file="/home/shailyjb/anlp-figlang/data/validation/en.csv",
        ).get_dataset_dict()
    )
    print(
        MablDatasetDict(
            train_file="/home/shailyjb/anlp-figlang/data/train/en.csv",
            validation_file="/home/shailyjb/anlp-figlang/data/validation/en.csv",
        ).get_dataset_dict()["train"][0]
    )
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")
dataset = MablDatasetDict(
    train_file="/home/shailyjb/anlp-figlang/data/train/en.csv",
    validation_file="/home/shailyjb/anlp-figlang/data/validation/en.csv",
)
dataset.tokenize_dataset(tokenizer=tokenizer)
print(dataset.get_dataset_dict()["train"][0])
print(type(dataset.get_dataset_dict()["train"][0]))
for key in dataset.get_dataset_dict()["train"][0]:
    print(key)
    print(dataset.get_dataset_dict()["train"][0][key])
    print(type(dataset.get_dataset_dict()["train"][0][key]))
    if key == "input_ids":
        print(dataset.get_dataset_dict()["train"][0][key][0])
        print(type(dataset.get_dataset_dict()["train"][0][key][0]))
