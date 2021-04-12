from nlp import Dataset, DatasetDict
import pandas as pd
from config import max_length, label2id
from model import tokenizer
import os
import torch


def convert_to_stsb_features(example_batch):
    inputs = example_batch['content']
    features = tokenizer.batch_encode_plus(
        inputs, truncation=True, max_length=max_length, padding='max_length')

    features["labels"] = [label2id[i] for i in example_batch["sentiment"]]
    return features


# check if cache exists all 3 files
cache_location = "data/cached/"
features_dict = {}
extra_feature_dict = {}

def convert_to_features(dataset_dict, convert_func_dict):
    columns_dict = {
        "document": ['input_ids', 'attention_mask', 'labels'],
        "paragraph": ['input_ids', 'attention_mask', 'labels'],
        "sentence": ['input_ids', 'attention_mask', 'labels'],
    }
    features_dict = {}

    for task_name, dataset in dataset_dict.items():
        features_dict[task_name] = {}
        print(task_name)
        for phase, phase_dataset in dataset.items():
            features_dict[task_name][phase] = phase_dataset.map(
                convert_func_dict[task_name],
                batched=True,
                load_from_cache_file=False,
            )
            print(task_name, phase, len(phase_dataset),
                  len(features_dict[task_name][phase]))
            features_dict[task_name][phase].set_format(
                type="torch",
                columns=columns_dict[task_name],
            )
            print(task_name, phase, len(phase_dataset),
                  len(features_dict[task_name][phase]))
    return features_dict


if os.path.exists(cache_location+"cached_data_features.pt"):
    print("Loading features from files")
    features_dict = torch.load(cache_location+"cached_data_features.pt")
    extra_feature_dict = torch.load(cache_location+"cached_extra_data_features.pt")
else:

    sentinews_location = "data/preprocessed/sentinews/"

    df_document_slovene_train = pd.read_csv(
        sentinews_location+"SentiNews_document_train.tsv", sep="\t")
    df_document_slovene_valid = pd.read_csv(
        sentinews_location+"SentiNews_document_valid.tsv", sep="\t")
    df_document_slovene_test = pd.read_csv(
        sentinews_location+"SentiNews_document_test.tsv", sep="\t")

    df_document_croatian_train = pd.read_csv(
        sentinews_location+"HrSentiNews_document_train.tsv", sep="\t")
    df_document_croatian_valid = pd.read_csv(
        sentinews_location+"HrSentiNews_document_valid.tsv", sep="\t")
    df_document_croatian_test = pd.read_csv(
        sentinews_location+"HrSentiNews_document_test.tsv", sep="\t")

    df_paragraph_train = pd.read_csv(
        sentinews_location+"SentiNews_paragraph_train.tsv", sep="\t")
    df_paragraph_valid = pd.read_csv(
        sentinews_location+"SentiNews_paragraph_valid.tsv", sep="\t")
    df_paragraph_test = pd.read_csv(
        sentinews_location+"SentiNews_paragraph_test.tsv", sep="\t")

    df_sentence_train = pd.read_csv(
        sentinews_location+"SentiNews_sentence_train.tsv", sep="\t")
    df_sentence_valid = pd.read_csv(
        sentinews_location+"SentiNews_sentence_valid.tsv", sep="\t")
    df_sentence_test = pd.read_csv(
        sentinews_location+"SentiNews_sentence_test.tsv", sep="\t")

    df_document_sl_hr_train = pd.read_csv(
        sentinews_location+"HRSLSentiNews_document_train.tsv", sep="\t")
    df_document_sl_hr_valid = pd.read_csv(
        sentinews_location+"HRSLSentiNews_document_valid.tsv", sep="\t")

    # NO test hr mixed as HR test will be used as final test

    # gather everyone if you want to have a single DatasetDict
    document = DatasetDict({
        "train": Dataset.from_pandas(df_document_sl_hr_train),
        "valid": Dataset.from_pandas(df_document_sl_hr_valid),
        "test": Dataset.from_pandas(df_document_croatian_test)
    })
    # document.save_to_disk("sentinews-document")
    # gather everyone if you want to have a single DatasetDict
    paragraph = DatasetDict({
        "train":  Dataset.from_pandas(df_paragraph_train),
        "valid": Dataset.from_pandas(df_paragraph_valid),
        "test":  Dataset.from_pandas(df_paragraph_test),
    })
    # paragraph.save_to_disk("sentinews-paragraph")
    # gather everyone if you want to have a single DatasetDict
    sentence = DatasetDict({
        "train": Dataset.from_pandas(df_sentence_train),
        "valid": Dataset.from_pandas(df_sentence_valid),
        "test": Dataset.from_pandas(df_sentence_test)

    })
    sl_document = DatasetDict({
        "test": Dataset.from_pandas(df_document_slovene_test)
    })

    dataset_dict = {
        "document": document,
        "paragraph": paragraph,
        "sentence": sentence,
    }

    extra_dataset_dict = {
        "document": sl_document,
    }

    print("labels :", df_document_sl_hr_train.sentiment.unique())

    for task_name, dataset in dataset_dict.items():
        print(task_name)
        print(dataset_dict[task_name]["train"][0])
        print()

    for task_name, dataset in extra_dataset_dict.items():
        print(task_name)
        print(extra_dataset_dict[task_name]["test"][0])
        print()

    convert_func_dict = {
        "document": convert_to_stsb_features,
        "paragraph": convert_to_stsb_features,
        "sentence": convert_to_stsb_features,
    }

    features_dict = convert_to_features(dataset_dict, convert_func_dict)

    extra_feature_dict = convert_to_features(extra_dataset_dict, {
        "document": convert_to_stsb_features
    })

    torch.save(features_dict, cache_location + 'cached_data_features.pt')
    torch.save(extra_feature_dict, cache_location +
               'cached_extra_data_features.pt')

train_dataset = {
    task_name: dataset["train"]
    for task_name, dataset in features_dict.items()
}

validation_dataset = {
    task_name: dataset["valid"]
    for task_name, dataset in features_dict.items()
}

extra_test_dataset ={
    task_name: dataset["test"]
    for task_name, dataset in extra_feature_dict.items()
}
