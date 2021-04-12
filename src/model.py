import transformers
from mtm import MultitaskModel
from config import model_name, drop_out

multitask_model = MultitaskModel.create(
    model_name=model_name,
    model_type_dict={
        "document": transformers.AutoModelForSequenceClassification,
        "paragraph": transformers.AutoModelForSequenceClassification,
        "sentence": transformers.AutoModelForSequenceClassification,
    },
    model_config_dict={
        "document": transformers.AutoConfig.from_pretrained(model_name, num_labels=3, hidden_dropout_prob=drop_out, attention_probs_dropout_prob=drop_out),
        "paragraph": transformers.AutoConfig.from_pretrained(model_name, num_labels=3, hidden_dropout_prob=drop_out, attention_probs_dropout_prob=drop_out),
        "sentence": transformers.AutoConfig.from_pretrained(model_name, num_labels=3, hidden_dropout_prob=drop_out, attention_probs_dropout_prob=drop_out),
    },
)

tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
