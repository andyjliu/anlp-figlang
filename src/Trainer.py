import argparse
import os
import random

import evaluate
import numpy as np
from DataCollator import DataCollatorForMultipleChoice
from MablDatasetDict import MablDatasetDict
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

import wandb

HF_MODEL_NAME_XLMR = "xlm-roberta-large"
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_prediction_label_tuples):
    predictions, labels = eval_prediction_label_tuples

    predictions = np.argmax(predictions, axis=1)

    return accuracy.compute(predictions=predictions, references=labels)


class CustomTrainer:
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
            compute_metrics=compute_metrics,
        )

    def train(self):
        self.trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=HF_MODEL_NAME_XLMR)
    parser.add_argument("--output-dir", type=str, default="output")
    parser.add_argument("--train-file", type=str, default="../data/train/en.csv")
    parser.add_argument("--val-file", type=str, default="../data/validation/en.csv")

    return parser.parse_args()


def get_model(model_name):
    return AutoModelForMultipleChoice.from_pretrained(model_name)


def configure_wandb():
    os.environ["WANDB_PROJECT"] = "anlp-figlang"
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mabl_dataset_dict = MablDatasetDict(train_file=args.train_file, validation_file=args.val_file)
    mabl_dataset_dict.tokenize_dataset(tokenizer)
    data_collator = DataCollatorForMultipleChoice(
        tokenizer, padding=True, pad_to_multiple_of=8
    )
    metric = evaluate.load("accuracy")
    train_dataset = mabl_dataset_dict.get_dataset_dict()["train"]
    validation_dataset = mabl_dataset_dict.get_dataset_dict()["validation"]

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    configure_wandb()

    trainer = CustomTrainer(
        model=get_model(args.model),
        output_dir=args.output_dir,
        batch_size=32,
        learning_rate=5e-6,
        num_training_epochs=20,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=data_collator,
    )

    trainer.train()

    wandb.finish()
