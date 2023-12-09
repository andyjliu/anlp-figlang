import argparse
import os
import random

import evaluate
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import (
    DataCollatorWithPadding,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

import wandb

MT5_SMALL = "google/mt5-small"
accuracy = evaluate.load("accuracy")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=MT5_SMALL)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/data/tir/projects/tir5/users/shailyjb/anlp-figlang/mt5_small_trial_2_epochs",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/home/shailyjb/anlp-figlang/data/train/en.csv",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="/home/shailyjb/anlp-figlang/data/validation/en.csv",
    )

    return parser.parse_args()


def configure_wandb():
    os.environ["WANDB_PROJECT"] = "anlp-figlang"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"


def get_model(model_name):
    return T5ForConditionalGeneration.from_pretrained(model_name)


class T5Trainer:
    def __init__(
        self,
        model,
        output_dir,
        batch_size,
        learning_rate,
        num_training_epochs,
        train_dataset,
        eval_dataset,
        data_collator,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            learning_rate=learning_rate,
            num_train_epochs=num_training_epochs,
            load_best_model_at_end=False,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to=["wandb"],
            logging_strategy="epoch",
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )

    def train(self):
        self.trainer.train()

    def compute_metrics(self, eval_prediction_label_tuples):
        predictions, labels = eval_prediction_label_tuples

        predictions = np.argmax(predictions, axis=1)

        return accuracy.compute(predictions=predictions, references=labels)


class DatasetForT5(DatasetDict):
    def __init__(self, train_file, validation_file):
        super().__init__()
        self.dataset_dict = {}

        self.dataset_dict["train"] = Dataset.from_csv(train_file)
        self.dataset_dict["validation"] = Dataset.from_csv(validation_file)

    def get_dataset_dict(self):
        return self.dataset_dict

    def prepare_input(self, tokenizer):
        def _prepare_example(example):
            # Create input by concatenating question and options
            metaphor = example["startphrase"]
            meaning_0 = example["ending1"]
            meaning_1 = example["ending2"]

            # Ensuring they all have full stops for uniformity.
            if metaphor[-1] != ".":
                metaphor += "."
            if meaning_0[-1] != ".":
                meaning_0 += "."
            if meaning_1[-1] != ".":
                meaning_1 += "."

            # Create prompt that will be input to the model
            input_text = f"metaphor: {metaphor} meanings: 0) {meaning_0} 1) {meaning_1}"

            # Tokenize input text
            tokenized_input = tokenizer.encode(
                input_text, return_tensors="pt", max_length=256, truncation=True
            )

            return {"input_ids": tokenized_input[0], "labels": example["labels"]}

        # tokenized_datasets = {}

        for key in self.dataset_dict:
            dataset = self.dataset_dict[key]
            # tokenized_dataset = dataset.map(_prepare_example)
            # tokenized_datasets[f"processed_{key}"] = tokenized_dataset
            self.dataset_dict[key] = dataset.map(
                _prepare_example, remove_columns=dataset.column_names, batched=False
            )

        # for key in tokenized_datasets:
        #     self.dataset_dict[key] = tokenized_datasets[key]


if __name__ == "__main__":
    args = parse_args()
    configure_wandb()

    dataset = DatasetForT5(
        train_file=args.train_file,
        validation_file=args.val_file,
    )
    # print(dataset.get_dataset_dict())
    # print(dataset.get_dataset_dict()["train"][0])
    tokenizer = T5Tokenizer.from_pretrained(MT5_SMALL)
    dataset.prepare_input(tokenizer)
    # print(dataset.get_dataset_dict())
    # print(dataset.get_dataset_dict()["train"][0])
    # print(dataset.get_dataset_dict()["train"][1])
    # print(dataset.get_dataset_dict()["train"][2])
    # print(dataset.get_dataset_dict()["train"][3])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    metric = evaluate.load("accuracy")

    train_dataset = dataset.get_dataset_dict()["train"]
    validation_dataset = dataset.get_dataset_dict()["validation"]

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    trainer = T5Trainer(
        model=get_model(args.model),
        output_dir=args.output_dir,
        batch_size=32,
        learning_rate=5e-3,
        num_training_epochs=2,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    wandb.finish()
