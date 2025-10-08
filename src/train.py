import os
from dataclasses import dataclass
from typing import Dict, Any

from datasets import load_dataset
from transformers import (
    AutoImageProcessor, AutoModelForImageClassification,
    TrainingArguments, Trainer
)
import numpy as np
from PIL import Image

MODEL_NAME = "google/vit-base-patch16-224-in21k"  # base; swap to your fb/vit if you used it
DATA_DIR = os.environ.get("DATA_DIR", "data")     # expects data/train, data/val with class folders
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "final_model")
SEED = 42

def load_folder_dataset(data_dir: str):
    # expects ImageFolder-style structure
    ds = load_dataset("imagefolder", data_dir=data_dir)  # splits if data/train, data/val present
    return ds

def main():
    ds = load_folder_dataset(DATA_DIR)

    # Label names from dataset
    labels = ds["train"].features["label"].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)

    def transform(example):
        image = example["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        enc = processor(images=image, return_tensors="pt")
        example["pixel_values"] = enc["pixel_values"][0]
        return example

    ds = ds.with_transform(transform)

    def collate_fn(batch):
        pixel_values = [x["pixel_values"] for x in batch]
        labels_batch = [x["label"] for x in batch]
        return {
            "pixel_values": np.stack([pv.numpy() for pv in pixel_values]),
            "labels": np.array(labels_batch),
        }

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # handy when swapping head sizes
    )

    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=5e-5,
        num_train_epochs=5,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=SEED,
        report_to="none",
        fp16=True
    )

    import evaluate
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(axis=-1)
        return {"accuracy": metric.compute(predictions=preds, references=labels)["accuracy"]}

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds.get("validation") or ds.get("val") or ds["train"].select(range(0, min(1000, len(ds["train"])))),
        compute_metrics=compute_metrics,
        tokenizer=processor,           # lets HF save preprocessor_config.json
        data_collator=collate_fn,
    )

    trainer.train()
    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)  # saves preprocessor_config.json

    # quick eval
    eval_out = trainer.evaluate()
    print("Final eval:", eval_out)

if __name__ == "__main__":
    main()
