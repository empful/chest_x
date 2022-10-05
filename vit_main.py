import torch
import os
import shutil
import numpy as np
import pandas as pd
import contextlib
import io
from pathlib import Path
from scipy.special import softmax
import json
import matplotlib.pyplot as plt


from torchvision import transforms
import transformers
from transformers import ViTConfig, TrainingArguments, Trainer
import datasets
from datasets import Image, list_metrics
import sklearn.metrics as sklm
from scipy.special import expit

from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)

from sklearn.metrics import classification_report


label_mapping = {'Cardiomegaly': 0, 'Emphysema': 1, 'Effusion': 2, 'Hernia': 3, 'Infiltration': 4, 'Mass': 5, 'Nodule': 6, 'Atelectasis': 7, 'Pneumothorax': 8, 'Pleural_Thickening': 9, 'Pneumonia': 10, 'Fibrosis': 11, 'Edema': 12, 'Consolidation': 13}
id2label = {label:id for id,label in label_mapping.items()}

class XRayTransform:
    """
    Transforms for pre-processing XRay data across a batch.
    """
    def __init__(self,feature_extractor):
        self.feature_extractor = feature_extractor
        self.transforms = transforms.Compose([
            transforms.Lambda(lambda pil_img: pil_img.convert("RGB")),
            transforms.Resize(feature_extractor.size),
            transforms.ToTensor(),
            transforms.Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std),
        ])

    def __call__(self, example_batch):
        example_batch["pixel_values"] = [self.transforms(pil_img) for pil_img in example_batch["Image"]]   
        # inputs = self.feature_extractor([x for x in example_batch['image']], return_tensors='pt')
        return example_batch

def vit_data_collator(batch):
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def column(matrix, i):
    return [row[i] for row in matrix]

# metric is auc
def compute_metrics(p):
    auc_dict={}
    target = p.label_ids #
    for name, id in label_mapping.items():
        target = column(p.label_ids,id)
        pred = expit(column(p.predictions,id))
        try:
            auc = sklm.roc_auc_score(np.array(target), np.array(pred))
            auc_dict[name] = auc
        except BaseException:
            print("can't calculate auc for " + name)
    auc_dict['Average_auc'] = np.mean(list(auc_dict.values()))
    return auc_dict



def main():

    data_files = {"train": "images/train.jsonl", "validation": "images/val.jsonl","test":"images/test.jsonl"}
    dataset = datasets.load_dataset("json", data_files=data_files).cast_column("Image", Image())
    print(dataset)

    # checkpoint = "/mnt/bd/medai-cv/cvpr/CXR8/vit-b-16-3/checkpoint-1050"
    pretrained_model_name_or_path = "google/vit-base-patch16-224-in21k"
    feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
    pretrained_model_name_or_path
    )



    normalize = Normalize(mean=feature_extractor.image_mean, std=feature_extractor.image_std)
    _train_transforms = Compose(
            [
                RandomResizedCrop(feature_extractor.size),
                RandomHorizontalFlip(),
                ToTensor(),
                normalize,
            ]
        )

    _val_transforms = Compose(
            [
                Resize(feature_extractor.size),
                CenterCrop(feature_extractor.size),
                ToTensor(),
                normalize,
            ]
        )

    def train_transforms(examples):
        examples['pixel_values'] = [_train_transforms(image.convert("RGB")) for image in examples['Image']]
        return examples

    def val_transforms(examples):
        examples['pixel_values'] = [_val_transforms(image.convert("RGB")) for image in examples['Image']]
        return examples


    # Set the training transforms
    dataset["train"].set_transform(train_transforms)
    dataset["validation"].set_transform(val_transforms)
    dataset["test"].set_transform(val_transforms)

    
    #pretrained_model_name_or_path
    model = transformers.AutoModelForImageClassification.from_pretrained(
        pretrained_model_name_or_path,
        num_labels=14,
        id2label=id2label,
        label2id=label_mapping
    )

    #set config and trainer
    training_args = TrainingArguments(
        output_dir="./vit-b-16-3",
        per_device_train_batch_size=128,
        per_device_eval_batch_size=32,
        evaluation_strategy="steps",
        num_train_epochs=10,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-4,
        save_total_limit=2,
        lr_scheduler_type="cosine",
        remove_unused_columns=False,
        metric_for_best_model="Average_auc",
        push_to_hub=False,
        report_to='tensorboard',
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=vit_data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        tokenizer=feature_extractor,
    )

    # load checkpoint
    # last_checkpoint = None
    # if os.path.isdir(training_args.output_dir) and not training_args.overwrite_output_dir:
    #     last_checkpoint = transformers.trainer_utils.get_last_checkpoint(training_args.output_dir)
    
    # train model
    train_results = trainer.train()
    trainer.save_model()
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    print(train_results)

    #run evaluataion

    metrics_val = trainer.evaluate(dataset["validation"])
    trainer.log_metrics("val", metrics_val)
    trainer.save_metrics("val", metrics_val)

    metrics_train = trainer.evaluate(dataset["train"])
    trainer.log_metrics("train", metrics_train)
    trainer.save_metrics("train", metrics_train)

    metrics_test = trainer.evaluate(dataset["test"])
    trainer.log_metrics("test", metrics_test)
    trainer.save_metrics("test", metrics_test)


main()

# sudo python3 -m torch.distributed.launch --nproc_per_node=2 vit_main.py --sharded_dpp