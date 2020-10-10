import sys
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import tensorflow as tf
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertForSequenceClassification
from transformers import TFTrainer
from transformers import TFTrainingArguments

# Categories of the given dataset
CATEGORIES = {'0': 'exploration',
              '1': 'headhunters',
              '2': 'intelligence',
              '3': 'logistics',
              '4': 'politics',
              '5': 'transportation',
              '6': 'weapons'}


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


def create_dataset(dataset_path: str):
    """Generates texts and labels for each data example

    Args:
        dataset_path (str): Path where the dataset is located.

    Returns:
        texts (list: str): Texts for each data example.
        labels (list: str): Label for each data example.

    """
    assert Path(dataset_path).is_dir(), f"Path {dataset_path} not valid"
    split_dir = Path(dataset_path)
    texts = []
    labels = []
    for label, category in CATEGORIES.items():
        for text_file in Path(f"{split_dir}/{category}").glob('*'):
            texts.append(text_file.read_text())
            labels.append(int(label))

    return texts, labels


def main():
    train_texts, train_labels = create_dataset(sys.argv[1])

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_texts, train_labels, test_size=.2)

    tokenizer = DistilBertTokenizerFast.from_pretrained(
        'distilbert-base-uncased', num_labels=len(CATEGORIES))

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    train_dataset = tf.data.Dataset.from_tensor_slices((
        dict(train_encodings),
        train_labels
    ))
    val_dataset = tf.data.Dataset.from_tensor_slices((
        dict(val_encodings),
        val_labels
    ))

    training_args = TFTrainingArguments(
        output_dir='./results',         # output directory
        num_train_epochs=10,            # total number of training epochs
        per_device_train_batch_size=8,  # batch size per device during training
        per_device_eval_batch_size=64,  # batch size for evaluation
        warmup_steps=500,               # number of warmup steps for learning rate scheduler
        weight_decay=0.01,              # strength of weight decay
        logging_dir='./logs',           # directory for storing logs
        logging_steps=10
    )

    with training_args.strategy.scope():
        model = TFDistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=len(CATEGORIES))

    trainer = TFTrainer(
        model=model,                    # the instantiated 🤗 Transformers model to be trained
        args=training_args,             # training arguments, defined above
        train_dataset=train_dataset,    # training dataset
        eval_dataset=val_dataset,       # evaluation dataset
        compute_metrics=compute_metrics
    )

    trainer.train()

    tokenizer.save_pretrained("models/tf_model")
    trainer.save_model('models/tf_model')


if __name__ == '__main__':
    main()




