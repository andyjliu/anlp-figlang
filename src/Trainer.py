import argparse
import os
import random

import evaluate
from src.DataCollator import DataCollatorForMultipleChoice
from src.MablDatasetDict import MablDatasetDict
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMultipleChoice,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

HF_MODEL_NAME_XLMR = "xlm-roberta-large"


class MyTrainer:
    def __init__(
        self,
        model,
        output_dir,
        batch_size,
        learning_rate,
        num_training_epochs,
        train_dataset,
        eval_dataset,
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
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            greater_is_better=False,
            report_to=["wandb"],
        )

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    def train(self):
        self.trainer.train()


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, default=HF_MODEL_NAME_XLMR)
    parser.add_argument("--output-dir", type=str, default="output")

    return parser.parse_args()


def get_model(model_name):
    return AutoModelForMultipleChoice.from_pretrained(model_name)


def configure_wandb():
    os.environ["WANDB_PROJECT"] = "anlp-figlang"
    os.environ["WANDB_LOG_MODEL"] = "true"
    os.environ["WANDB_WATCH"] = "false"


if __name__ == "__main__":
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    mabl_dataset_dict = MablDatasetDict(data_dir="../data")
    mabl_dataset_dict.tokenize_dataset(tokenizer)
    # print(mabl_dataset_dict.get_dataset_dict())
    data_collator = DataCollatorForMultipleChoice(tokenizer, pad_to_multiple_of=8)
    metric = evaluate.load("accuracy")
    train_dataset = mabl_dataset_dict.get_dataset_dict()["train"]
    validation_dataset = mabl_dataset_dict.get_dataset_dict()["validation"]

    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=32
    )
    eval_dataloader = DataLoader(
        validation_dataset,
        collate_fn=data_collator,
        batch_size=args["per_device_eval_batch_size"],
    )

    configure_wandb()

    trainer = MyTrainer(
        model=get_model(args.model),
        output_dir=args.output_dir,
        batch_size=32,
        learning_rate=5e-6,
        num_training_epochs=20,
        train_dataset=mabl_dataset_dict.get_dataset_dict()["train"],
        eval_dataset=mabl_dataset_dict.get_dataset_dict()["validation"],
    )

    trainer.train()
