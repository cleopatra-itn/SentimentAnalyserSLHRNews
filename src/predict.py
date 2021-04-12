import datasets
import numpy as np
import torch
import transformers
from config import epochs, batch_size, learning_rate
from model import tokenizer, multitask_model
from mtm import MultitaskTrainer, NLPDataCollator, DataLoaderWithTaskname

from data import features_dict

##UPDATE THIS
multitask_model.load_state_dict(torch.load(
    "src/models/{}/pytorch_model.bin"))

trainer = MultitaskTrainer(
    model=multitask_model,
    args=transformers.TrainingArguments(
        learning_rate=learning_rate,
        output_dir="/tmp",
        do_train=False,
        do_eval=True,
        # evaluation_strategy ="steps",
        num_train_epochs=epochs,
        fp16=True,
        # Adjust batch size if this doesn't fit on the Colab GPU
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        save_steps=3000,
        # eval_steps=50,
        load_best_model_at_end=True,
    ),
    data_collator=NLPDataCollator(tokenizer=tokenizer),
    callbacks=[]
)

tests_dict = {}
for task_name in ["document", "paragraph", "sentence"]:
    test_dataloader = DataLoaderWithTaskname(
        task_name,
        trainer.get_eval_dataloader(features_dict[task_name]["test"])
    )
    print(test_dataloader.data_loader.collate_fn)
    tests_dict[task_name] = trainer.prediction_loop(
        test_dataloader,
        description=f"Testing: {task_name}"
    )

for task_name in ["document", "paragraph", "sentence"]:
    for metric in ["precision", "recall", "f1"]:
        print("test {} {}:".format(metric, task_name),
              datasets.load_metric(metric,
                                   name="dev {} {}".format(metric, task_name)).compute(
                  predictions=np.argmax(
                      tests_dict[task_name].predictions, axis=1),
                  references=tests_dict[task_name].label_ids, average="macro"
              ))
