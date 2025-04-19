import os
from collections import Counter
from types import SimpleNamespace

import numpy as np
from scipy import stats
from sudachipy import SplitMode

from ln_recommender.text import Epub


def get_freq(filename):
    freq_dict = {}
    with open(filename, encoding="utf8") as fd:
        for count, line in enumerate(fd):
            stripped_line = line.strip()
            if freq_dict.get(stripped_line) is None:
                freq_dict[stripped_line] = count
    return freq_dict


def parse_text(path, filename, freq, tokenizer_obj):
    verb_string = "動詞"
    aux_string = "助動詞"
    verb = []
    aux = []

    skip_tokens = ["補助記号", "空白", "数詞"]
    suda_freq = Counter({})
    suda_avg_line_length = 0
    text_line_length = []

    def process(line):
        nonlocal suda_freq
        nonlocal suda_avg_line_length
        nonlocal text_line_length
        nonlocal verb
        nonlocal aux

        stripped_line = line.strip()
        if stripped_line == "":
            return

        tokens = tokenizer_obj.tokenize(stripped_line, SplitMode.A)

        stripped_line_len = len(stripped_line)
        sent = (
            stripped_line.count("」", stripped_line_len - 1)
            + stripped_line.count("』", stripped_line_len - 1)
            + stripped_line.count("。")
            + stripped_line.count("？")
            + stripped_line.count("！")
        )
        sent = (
            sent
            - stripped_line.count("。」", stripped_line_len - 2)
            - stripped_line.count("？」", stripped_line_len - 2)
            - stripped_line.count("！」", stripped_line_len - 2)
        )
        sent = (
            sent
            - stripped_line.count("。』", stripped_line_len - 2)
            - stripped_line.count("？』", stripped_line_len - 2)
            - stripped_line.count("！』", stripped_line_len - 2)
        )
        if sent == 0:
            sent = 1
        elif sent < 0:
            print("Negative line length")
            print(f"'{stripped_line}'")

        count_tokens = 0
        count_dict = Counter({})
        ca = 0
        cv = 0
        for w in tokens:
            if any(x in skip_tokens for x in w.part_of_speech()):
                continue
            elif all(ord(c) < 128 for c in w.dictionary_form()):
                continue
            else:
                count_dict[w.dictionary_form()] += 1
                count_tokens += 1
                if verb_string in w.part_of_speech():
                    cv += 1
                elif aux_string in w.part_of_speech():
                    ca += 1

        if count_tokens > 0:
            text_line_length.append(count_tokens / sent)
            verb.append(cv / sent)
            aux.append(ca / sent)
        suda_freq = suda_freq + count_dict

    if os.path.splitext(filename)[1] == ".epub":
        epub = Epub.from_file(path)
        for p in epub.text():
            process(p.text())
    else:
        with open(path, "r", encoding="utf8") as fd:
            for line in fd:
                process(line)

    text_line_length = np.array(text_line_length)
    suda_avg_line_length = np.mean(text_line_length)
    suda_median_line_length = np.median(text_line_length)

    m = stats.mode(text_line_length)
    suda_mode_line_length = m[0]

    dict_freq = calculate_freq(freq, suda_freq)
    suda_verb = np.mean(verb)
    suda_aux = np.mean(aux)

    return SimpleNamespace(
        filename=filename,
        p70=np.percentile(dict_freq, 70),
        p80=np.percentile(dict_freq, 80),
        p90=np.percentile(dict_freq, 90),
        p95=np.percentile(dict_freq, 95),
        p99=np.percentile(dict_freq, 99),
        avg=suda_avg_line_length,
        median=suda_median_line_length,
        mode=suda_mode_line_length,
        verb=suda_verb,
        aux=suda_aux,
    )


def calculate_freq(freq_dict, suda_freq):
    dict_freq = []
    for w, c in suda_freq.most_common():
        fq = freq_dict.get(w)
        if fq is not None:
            for i in range(c):
                dict_freq.append(fq)

    return np.array(dict_freq)
