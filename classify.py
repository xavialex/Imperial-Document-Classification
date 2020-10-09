"""Text classification.

CLI script that expects the following arguments:
    python classify.py model_path file1 file2 file3 ...
It'll process any number of files with the model located in model_path

"""

import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification


CATEGORIES = {'0': 'exploration',
              '1': 'headhunters',
              '2': 'intelligence',
              '3': 'logistics',
              '4': 'politics',
              '5': 'transportation',
              '6': 'weapons'}


def read_txt_file(txt_path: str) -> str:
    assert Path(txt_path).exists(), f"File '{txt_path}' not found"
    return Path(txt_path).read_text()


def main():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
    model = TFAutoModelForSequenceClassification.from_pretrained(sys.argv[1])
    # Print filename and identified class for every file passed
    for txt_file in sys.argv[2:]:
        text = read_txt_file(txt_file)
        model_output = model(tokenizer([text], padding=True, truncation=True,
                                       return_tensors='tf'))
        label = str(np.argmax(model_output[0]))
        label_name = CATEGORIES[label]
        print(f"{txt_file} {label_name}")


if __name__ == '__main__':
    main()
