import logging
import datetime
from typing import Dict, List, Union

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import Dataset, DatasetDict

from config import drop_out, epochs, output_path
from data import (features_dict, train_dataset,
                  extra_feature_dict, validation_dataset)
from model import multitask_model, tokenizer
from mtm import (DataLoaderWithTaskname, MultitaskTrainer,
                 NLPDataCollator)
from utils import get_timestamp

logging.basicConfig(level=logging.ERROR)
torch.manual_seed(42)


trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        output_dir="./models/multitask_model",
        overwrite_output_dir=True,
        learning_rate=2e-5,
        do_train=True,
        do_eval=True,
        # evaluation_strategy ="steps",
        num_train_epochs=epochs,
        fp16=True,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        save_steps=3000,
        # eval_steps=50,
        load_best_model_at_end=True,
    ),
    data_collator=NLPDataCollator(tokenizer=tokenizer),
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    callbacks=[]
)
# train the model
trainer.train()

ts = datetime.datetime.now()
trainer.save_model("./models/multitask_model_{}".format(ts))

preds_dict = {}
for task_name in ["document", "paragraph", "sentence"]:
    if task_name in features_dict:
        eval_dataloader = DataLoaderWithTaskname(
            task_name,
            trainer.get_eval_dataloader(features_dict[task_name]["valid"],)
        )
        print(eval_dataloader.data_loader.collate_fn)
        preds_dict[task_name] = trainer.prediction_loop(
            eval_dataloader,
            description=f"Validation: {task_name}"
        )

tests_dict = {}
for task_name in ["document", "paragraph", "sentence"]:
    if task_name in features_dict:
        test_dataloader = DataLoaderWithTaskname(
            task_name,
            trainer.get_eval_dataloader(features_dict[task_name]["test"])
        )
        print(test_dataloader.data_loader.collate_fn)
        tests_dict[task_name] = trainer.prediction_loop(
            test_dataloader,
            description=f"Testing: {task_name}"
        )
torch.save(tests_dict, output_path+"exp1-test_{}".format(ts))
extra_tests_dict = {}
for task_name in ["document"]:
    test_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(extra_feature_dict[task_name]["test"])
    )
    print(test_dataloader.data_loader.collate_fn)
    extra_tests_dict[task_name] = trainer.prediction_loop(
        test_dataloader,
        description=f"Testing Extra Test: {task_name}"
    )
torch.save(extra_tests_dict, output_path+"exp1-test_extra_{}".format(ts))

with open("log.out", "a") as outputfile:
    print('-----------epoch=', epochs, get_timestamp(),
          "---------------------------", file=outputfile)
    print("dropout:", drop_out, file=outputfile)
    print("dev loss d",
          preds_dict['document'].metrics["eval_loss"], file=outputfile)
    print("test loss d",
          tests_dict['document'].metrics["eval_loss"], file=outputfile)
    if 'paragraph' in preds_dict:
        print("dev loss p",
              preds_dict['paragraph'].metrics["eval_loss"], file=outputfile)
        print("test loss p",
              tests_dict['paragraph'].metrics["eval_loss"], file=outputfile)
    if 'sentence' in preds_dict:
        print("dev loss s",
              preds_dict['sentence'].metrics["eval_loss"], file=outputfile)
        print("test loss s",
              tests_dict['sentence'].metrics["eval_loss"], file=outputfile)

    print("===:loss:===", file=outputfile)
    for task_name in ["document", "paragraph", "sentence"]:
        for metric in ["precision", "recall", "f1"]:
            if task_name in preds_dict:
                print("dev {} {}:".format(metric, task_name),
                      datasets.load_metric(metric,
                                           name="dev {} {}".format(metric, task_name)).compute(
                    predictions=np.argmax(
                        preds_dict[task_name].predictions, axis=1),
                    references=preds_dict[task_name].label_ids, average="macro"
                ), file=outputfile)

    for task_name in ["document", "paragraph", "sentence"]:
        for metric in ["precision", "recall", "f1"]:
            if task_name in tests_dict:
                print("test {} {}:".format(metric, task_name),
                      datasets.load_metric(metric,
                                           name="test {} {}".format(metric, task_name)).compute(
                    predictions=np.argmax(
                        tests_dict[task_name].predictions, axis=1),
                    references=tests_dict[task_name].label_ids, average="macro"
                ), file=outputfile)
    for metric in ["precision", "recall", "f1"]:
        for task_name in ["document"]:
            print("extra test {} {}:".format(metric, task_name),
                  datasets.load_metric(metric,
                                       name="extra test {} {}".format(metric, task_name)).compute(
                predictions=np.argmax(
                    extra_tests_dict[task_name].predictions, axis=1),
                references=extra_tests_dict[task_name].label_ids, average="macro"
            ), file=outputfile)

    if "document" in preds_dict:
        print("dev accuracy document", datasets.load_metric('accuracy', name="dev accuracy document").compute(
            predictions=np.argmax(preds_dict["document"].predictions, axis=1),
            references=preds_dict["document"].label_ids,
        ), file=outputfile)
    if "paragraph" in preds_dict:
        print("dev accuracy paragraph", datasets.load_metric('accuracy', name="dev accuracy paragraph").compute(
            predictions=np.argmax(preds_dict["paragraph"].predictions, axis=1),
            references=preds_dict["paragraph"].label_ids,
        ), file=outputfile)
    if "sentence" in preds_dict:
        print("dev accuracy sentence", datasets.load_metric('accuracy', name="dev accuracy sentence").compute(
            predictions=np.argmax(preds_dict["sentence"].predictions, axis=1),
            references=preds_dict["sentence"].label_ids,
        ), file=outputfile)

    if 'document' in preds_dict:
        print("dev f1 document", datasets.load_metric('f1', name="dev f1 document").compute(
            predictions=np.argmax(preds_dict["document"].predictions, axis=1),
            references=preds_dict["document"].label_ids, average="macro"
        ), file=outputfile)
    if 'paragraph' in preds_dict:
        print("dev f1 paragraph", datasets.load_metric('f1', name="dev f1 paragraph").compute(
            predictions=np.argmax(preds_dict["paragraph"].predictions, axis=1),
            references=preds_dict["paragraph"].label_ids, average="macro"
        ), file=outputfile)
    if 'sentence' in preds_dict:
        print("dev f1 sentence", datasets.load_metric('f1',  name="dev f1 sentence").compute(
            predictions=np.argmax(preds_dict["sentence"].predictions, axis=1),
            references=preds_dict["sentence"].label_ids, average="macro"
        ), file=outputfile)

    # Test score
    if "document" in tests_dict:
        print("tad", datasets.load_metric('accuracy', name="test accuracy document").compute(
            predictions=np.argmax(tests_dict["document"].predictions, axis=1),
            references=tests_dict["document"].label_ids,
        ), file=outputfile)
    if "document" in tests_dict:
        print("tf1d", datasets.load_metric('f1', name="test f1 document").compute(
            predictions=np.argmax(tests_dict["document"].predictions, axis=1),
            references=tests_dict["document"].label_ids, average="macro"
        ), file=outputfile)

    if "sentence" in tests_dict:
        print("tas", datasets.load_metric('accuracy', name="test accuracy sentence").compute(
            predictions=np.argmax(tests_dict["sentence"].predictions, axis=1),
            references=tests_dict["sentence"].label_ids,
        ), file=outputfile)
    if "sentence" in tests_dict:
        print("tf1s", datasets.load_metric('f1',  name="test f1 sentence").compute(
            predictions=np.argmax(tests_dict["sentence"].predictions, axis=1),
            references=tests_dict["sentence"].label_ids, average="macro"
        ), file=outputfile)

    if 'paragraph' in tests_dict:
        print("tap", datasets.load_metric('accuracy', name="test accuracy paragraph").compute(
            predictions=np.argmax(tests_dict["paragraph"].predictions, axis=1),
            references=tests_dict["paragraph"].label_ids,
        ), file=outputfile)
    if 'paragraph' in tests_dict:
        print("tf1p", datasets.load_metric('f1', name="test f1 paragraph").compute(
            predictions=np.argmax(tests_dict["paragraph"].predictions, axis=1),
            references=tests_dict["paragraph"].label_ids, average="macro"
        ), file=outputfile)
