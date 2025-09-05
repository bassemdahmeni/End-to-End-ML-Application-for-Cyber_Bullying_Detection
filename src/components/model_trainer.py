import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
import torch
import evaluate
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np


@dataclass
class ModelTrainerConfig:
    full_ft_artifact: str = os.path.join("artifacts", "full_ft_model")
    partial_ft_artifact: str = os.path.join("artifacts", "partial_ft_model")
    lora_artifact: str = os.path.join("artifacts", "lora_model")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        os.makedirs("artifacts", exist_ok=True)
        self.label2id = {"normal": 0, "offensive": 1, "hatespeech": 2}
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.model_name = "distilbert-base-uncased"

    # -----------------------------
    # Full Fine-Tuning
    # -----------------------------
    def train_full_ft(self, dataset):
        logging.info("Starting Full Fine-Tuning")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def preprocess(batch):
            return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)

        dataset_enc = dataset.map(preprocess, batched=True)
        dataset_enc = dataset_enc.remove_columns(["comment", "__index_level_0__"])
        dataset_enc.set_format("torch")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = logits.argmax(-1)
            return {
                "accuracy": accuracy_metric.compute(predictions=preds, references=labels)["accuracy"],
                "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
            }

        training_args = TrainingArguments(
            output_dir=self.config.full_ft_artifact,
            do_eval=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=4,
            weight_decay=0.01,
            logging_dir="./logs_full_ft",
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_enc["train"],
            eval_dataset=dataset_enc["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(self.config.full_ft_artifact)
        tokenizer.save_pretrained(self.config.full_ft_artifact)
        logging.info(f"Full Fine-Tuned Model saved to {self.config.full_ft_artifact}")

    # -----------------------------
    # Partial Fine-Tuning (PEFT)
    # -----------------------------
    def train_partial_ft(self, dataset):
        logging.info("Starting Partial Fine-Tuning")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        def preprocess(batch):
            return tokenizer(batch["comment"], truncation=True, padding="max_length", max_length=128)

        dataset_enc = dataset.map(preprocess, batched=True)
        dataset_enc = dataset_enc.remove_columns(["comment", "__index_level_0__"])
        dataset_enc.set_format("torch")

        # Explicit cast to long
        dataset_enc = dataset_enc.map(lambda x: {"labels": x["label"].to(torch.long)}, batched=True)
        dataset_enc = dataset_enc.remove_columns(["label"])
        dataset_enc = dataset_enc.rename_column("labels", "label")

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        model.to(device)

        # Freeze all layers first
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze classifier, pre-classifier, last transformer layer
        for param in model.classifier.parameters():
            param.requires_grad = True
        for param in model.pre_classifier.parameters():
            param.requires_grad = True
        for param in model.distilbert.transformer.layer[-1].parameters():
            param.requires_grad = True

        accuracy_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            preds = torch.argmax(torch.tensor(logits), dim=-1).cpu().numpy()
            acc = accuracy_metric.compute(predictions=preds, references=labels)
            f1_score = f1_metric.compute(predictions=preds, references=labels, average="weighted")
            return {"accuracy": acc["accuracy"], "f1": f1_score["f1"]}

        training_args = TrainingArguments(
            output_dir=self.config.partial_ft_artifact,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=7,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_dir="./logs_partial_ft",
            logging_strategy="steps",
            logging_steps=50,
            report_to="none",
            fp16=True
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_enc["train"],
            eval_dataset=dataset_enc["validation"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(self.config.partial_ft_artifact)
        tokenizer.save_pretrained(self.config.partial_ft_artifact)
        logging.info(f"Partial Fine-Tuned Model saved to {self.config.partial_ft_artifact}")

    # -----------------------------
    # LoRA Fine-Tuning
    # -----------------------------
    def train_lora(self, dataset):
        logging.info("Starting LoRA Fine-Tuning")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

        def preprocess(batch):
            tokenized = tokenizer(batch["comment"], truncation=True, padding=False, max_length=256)
            tokenized["labels"] = batch["label"]
            return tokenized

        dataset_enc = dataset.map(preprocess, batched=True)
        dataset_enc = dataset_enc.remove_columns(["comment", "__index_level_0__"])

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )

        # LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_lin", "v_lin", "k_lin", "out_lin"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, max_length=256)

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            from sklearn.metrics import accuracy_score, f1_score
            return {"accuracy": accuracy_score(labels, predictions),
                    "f1": f1_score(labels, predictions, average="weighted")}

        training_args = TrainingArguments(
            output_dir=self.config.lora_artifact,
            overwrite_output_dir=True,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=6,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_ratio=0.1,
            logging_dir="./logs_lora",
            logging_steps=50,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            dataloader_pin_memory=False,
            gradient_accumulation_steps=1,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset_enc["train"],
            eval_dataset=dataset_enc["validation"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics
        )

        trainer.train()
        trainer.save_model(self.config.lora_artifact)
        tokenizer.save_pretrained(self.config.lora_artifact)
        logging.info(f"LoRA Fine-Tuned Model saved to {self.config.lora_artifact}")


if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    # -----------------------------
    # Run Data Ingestion
    # -----------------------------
    ingestion = DataIngestion()
    train_path, val_path, test_path = ingestion.initiate_data_ingestion()

    # -----------------------------
    # Run Data Transformation
    # -----------------------------
    transformation = DataTransformation()
    dataset, label2id, id2label = transformation.initiate_data_transformation(train_path, test_path)

    # -----------------------------
    # Initialize Model Trainer
    # -----------------------------
    trainer = ModelTrainer()

    # ⚠️ You won’t actually run this. These lines would train the models:
    # trainer.train_full_ft(dataset)
    # trainer.train_partial_ft(dataset)
    # trainer.train_lora(dataset)

    print("✅ ModelTrainer is ready. Artifacts paths are configured in `artifacts/`")
