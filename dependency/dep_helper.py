import json
from collections import defaultdict


def get_vocab(train_data_path):
    vocab = set()
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            word = splits[1]
            vocab.add(word)
    return vocab


def get_word2id(train_data_path, do_lower_case=False, freq_threshold=2):
    word2count =defaultdict(int)
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            word = splits[1]
            if do_lower_case:
                word = word.lower()
            word2count[word] += 1

    word2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2}
    index = 3
    for word, count in word2count.items():
        if count >= freq_threshold:
            word2id[word] = index
            index += 1

    return word2id


def get_label_list(train_data_path):
    label_list = ['<UNK>']

    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            dep = splits[7]
            if dep not in label_list:
                label_list.append(dep)

    label_list.extend(['[CLS]', '[SEP]'])
    return label_list


def get_pos2id(train_data_path):
    pos2id = {'<PAD>': 0, '<UNK>': 1, '[CLS]': 2}
    index = 3
    with open(train_data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) == 0:
                continue
            splits = line.split('\t')
            pos = splits[3]
            if pos not in pos2id:
                pos2id[pos] = index
                index += 1
    return pos2id


def get_wordpair_list(data_path):
    wordpair2id = {}
    i = 1
    with open(data_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line == '':
                continue
            splits = line.split()
            wordpair2id[(splits[0], splits[1])] = i
            i += 1
    return wordpair2id


def save_json(file_path, data):
    with open(file_path, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False)
        f.write('\n')


def load_json(file_path):
    with open(file_path, 'r', encoding='utf8') as f:
        line = f.readline()
    return json.loads(line)
