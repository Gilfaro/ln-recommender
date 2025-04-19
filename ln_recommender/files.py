import fnmatch
import os

import pandas as pd

TEXT_FORMATS = ["epub", "txt"]

SUPPORTED_TEXT_FORMATS = ["*." + extension for extension in TEXT_FORMATS]

CSV_HEADERS = [
    "70%",
    "80%",
    "90%",
    "95%",
    "99%",
    "Avg Sentence Length",
    "Median Sentence Length",
    "Mode Sentence Length",
    "Avg Verb Count",
    "Avg Auxiliary Verb Count",
    "Label",
    "Comment",
]


def get_sources(input):
    files = []
    for dir in input.dirs:
        files.extend(get_sources_dir(dir))
    for text in input.text:
        files.append(os.path.split(text))
    return files


def get_sources_dir(directory):
    files = []
    for root, _, filenames in os.walk(directory):
        for file in filenames:
            for type in SUPPORTED_TEXT_FORMATS:
                if fnmatch.fnmatch(file, type):
                    files.append((root, file))
    return files


def save_csv(filename, data):
    df = pd.DataFrame(data, columns=CSV_HEADERS)
    df.to_csv(filename, sep=",", quotechar="'", index=False, float_format="%.2f")
