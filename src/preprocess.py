
# Mixed HR+SL dataset with 20% test
# Total Croatian dataset with 20% test => This is just for testing
# SL-Paragraph
# SL-Sentence
# News headline 20% with 0.5% dev
# clarin-twitter 20% with 0.5% dev

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
np.random.seed(42)


def print_statistics(df, label_col, name=""):
    print(name, " length", len(df))
    print(df[label_col].value_counts())


# get each dataset and its statistics and distribution
# Slovene Sentinews
sentinews_location = "data/sentinews/"
sentinews_output_location = "data/preprocessed/sentinews/"
required_columns = ['nid', 'content', 'sentiment']
# 10417
df_document = pd.read_csv(
    sentinews_location+"SentiNews_document-level.tsv", sep="\t")
print_statistics(df_document, "sentiment", "slovene document before duplicate")
df_document = df_document.drop_duplicates(subset=["content"])
df_document = df_document[df_document['content'].notna()]

df_document = df_document[required_columns]

print_statistics(df_document, "sentiment", "slovene document after duplicate")
train_valid, test = train_test_split(df_document,
                                     test_size=2000,
                                     random_state=42,
                                     stratify=df_document.sentiment)
train, valid = train_test_split(train_valid,
                                test_size=417,
                                random_state=42,
                                stratify=train_valid.sentiment)
print(len(train), len(valid), len(test))
train.to_csv(sentinews_output_location +
             "SentiNews_document_train.tsv", sep="\t", index=False)
valid.to_csv(sentinews_output_location +
             "SentiNews_document_valid.tsv", sep="\t", index=False)
test.to_csv(sentinews_output_location +
            "SentiNews_document_test.tsv", sep="\t", index=False)
# 1988
df_document_croatian = pd.read_csv(
    sentinews_location+"HrSentiNews_sentence-level.tsv", sep="\t")
print_statistics(df_document_croatian, "sentiment",
                 "croatian document before duplicate")
df_document_croatian = df_document_croatian.drop_duplicates(subset=["content"])
df_document_croatian = df_document_croatian[df_document_croatian['content'].notna(
)]


df_document_croatian = df_document_croatian[required_columns]
print_statistics(df_document_croatian, "sentiment",
                 "croatian document after duplicate")
train_valid, test = train_test_split(df_document_croatian,
                                     test_size=0.20,
                                     random_state=42,
                                     stratify=df_document_croatian.sentiment)
train, valid = train_test_split(train_valid,
                                test_size=200,
                                random_state=42,
                                stratify=train_valid.sentiment)
print("Croatian", len(train), len(valid), len(test))
train.to_csv(sentinews_output_location +
             "HrSentiNews_document_train.tsv", sep="\t", index=False)
valid.to_csv(sentinews_output_location +
             "HrSentiNews_document_valid.tsv", sep="\t", index=False)
test.to_csv(sentinews_output_location +
            "HrSentiNews_document_test.tsv", sep="\t", index=False)


# # 86803
df_paragraph = pd.read_csv(
    sentinews_location+"SentiNews_paragraph-level.tsv", sep="\t")
print_statistics(df_paragraph, "sentiment", "slovene para before duplicate")
# df_paragraph = df_paragraph.dropna()
df_paragraph = df_paragraph.drop_duplicates(subset=["content"])
df_paragraph = df_paragraph[df_paragraph['content'].notna()]
df_paragraph = df_paragraph[required_columns]
print_statistics(df_paragraph, "sentiment", "slovene para after duplicate")
train_valid, test = train_test_split(df_paragraph,
                                     test_size=0.20,
                                     random_state=42,
                                     stratify=df_paragraph.sentiment)
train, valid = train_test_split(train_valid,
                                test_size=0.10,
                                random_state=42,
                                stratify=train_valid.sentiment)
print("slovene para", len(train), len(valid), len(test))
train.to_csv(sentinews_output_location +
             "SentiNews_paragraph_train.tsv", sep="\t", index=False)
valid.to_csv(sentinews_output_location +
             "SentiNews_paragraph_valid.tsv", sep="\t", index=False)
test.to_csv(sentinews_output_location +
            "SentiNews_paragraph_test.tsv", sep="\t", index=False)

# # 161291
df_sentence = pd.read_csv(
    sentinews_location+"SentiNews_sentence-level.tsv", sep="\t")
print_statistics(df_sentence, "sentiment", "slovene sent before duplicate")
# df_sentence = df_sentence.dropna()
df_sentence = df_sentence.drop_duplicates(subset=["content"])
df_sentence = df_sentence[df_sentence['content'].notna()]
df_sentence = df_sentence[required_columns]

print_statistics(df_sentence, "sentiment", "slovene sent after duplicate")
train_valid, test= train_test_split(df_sentence,
                                     test_size = 0.20,
                                     random_state = 42,
                                     stratify = df_sentence.sentiment)
train, valid=train_test_split(train_valid,
                                test_size=5000,
                                random_state=42,
                                stratify=train_valid.sentiment)
print("slovene sent",len(train),len(valid),len(test)) 
train.to_csv(sentinews_output_location+"SentiNews_sentence_train.tsv",sep="\t",index=False)
valid.to_csv(sentinews_output_location+"SentiNews_sentence_valid.tsv",sep="\t",index=False)
test.to_csv(sentinews_output_location+"SentiNews_sentence_test.tsv",sep="\t",index=False)

# HR+SL train  HR+
df_doc_sl_train  = pd.read_csv(sentinews_output_location+"SentiNews_document_train.tsv",sep="\t")
df_doc_sl_val = pd.read_csv(sentinews_output_location+"SentiNews_document_valid.tsv",sep="\t")

df_doc_hr_train  = pd.read_csv(sentinews_output_location+"HrSentiNews_document_train.tsv",sep="\t")
df_doc_hr_val = pd.read_csv(sentinews_output_location+"HrSentiNews_document_valid.tsv",sep="\t")

df_doc_sl_train =  pd.concat([df_doc_hr_train, df_doc_sl_train])
df_doc_sl_train.to_csv(sentinews_output_location+"HRSLSentiNews_document_train.tsv",sep="\t",index=False)

df_doc_sl_val =  pd.concat([df_doc_hr_val, df_doc_sl_val])
df_doc_sl_train.to_csv(sentinews_output_location+"HRSLSentiNews_document_valid.tsv",sep="\t",index=False)
print(df_doc_sl_train.shape,df_doc_sl_val.shape)