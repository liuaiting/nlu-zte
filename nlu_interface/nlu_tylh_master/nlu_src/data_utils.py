# -*-coding=utf-8-*-
"""
Created on Thur Mar 2 2017

@author: Aiting Liu

Prepare data for multi-task RNN model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys

import tensorflow as tf

# Special vocabulary symbols - we always put them at the start.
_PAD = "_PAD"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _UNK]

START_VOCAB_dict = dict()
START_VOCAB_dict['with_padding'] = [_PAD, _UNK]
START_VOCAB_dict['no_padding'] = []

PAD_ID = 0

UNK_ID_dict = dict()
UNK_ID_dict['with_padding'] = 1  # sequence labeling (slot filling) need padding (mask)
# UNK_ID_dict['no_padding'] = 0  # sequence classification (intent detection)

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(，。！？、：；（）])")
_DIGIT_RE = re.compile(r"\d")


def basic_tokenizer(sentence):
    """Very basic tokenizer: split the sentence into a list of tokens."""
    words = []
    for space_sepatated_fragment in sentence.strip().split():
        words.extend(re.split(_WORD_SPLIT, space_sepatated_fragment))
    return [w for w in words if w]


def naive_tokenizer(sentence):
    """Naive tokenizer: split the sentence by space into a list of tokens."""
    return sentence.split()


def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size,
                      tokenizer=None, normalize_digits=False):
    """Create vocabulary file (if if does not exist yet) from data file.

    Data file is assumed to contain one sentence per line. Each sentence is
    tokenized and digits are normalized (if normalize_digits is set).
    Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
    We write it to vocabulary_path in a one-token-per-line format, so that later
    token in the first line gets id=0, second line gets id=1, and so on.

    Args:
        vocabulary_path: path where the vocabulary will be created.
        data_path: data file that will be used to created vocabulary.
        max_vocabulary_size: limit on the size of the created vocabulary.
        tokenizer: a function to use to tokenize each data sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="r") as f:
            counter = 0
            token_counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                tokens = tokenizer(line) if tokenizer else basic_tokenizer(line)
                token_counter += len(tokens)
                for w in tokens:
                    word = re.sub(_DIGIT_RE, "0", w) if normalize_digits else w
                    if word in vocab:
                        vocab[word] += 1
                    else:
                        vocab[word] = 1
            vocab_list = START_VOCAB_dict["with_padding"] + sorted(vocab, key=vocab.get, reverse=True)
            if len(vocab_list) > max_vocabulary_size:
                vocab_list = vocab_list[:max_vocabulary_size]
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for w in vocab_list:
                    vocab_file.write(w + "\n")
            print('There are %d lines in training data.' % counter)
            print('There are %d tokens in training data.' % token_counter)
            print('There are %d words in training data.' % len(vocab_list))


def initialize_vocabulary(vocabulary_path):
    """Initialize vocabulary from file.

    We assume the vocabulary is stored one-item-per-line, so a file:
        dog
        cat
    will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
    also return the reversed-vocabulary ["dog", "cat"].

    Args:
        vocabulary_path: path to the file containing the vocabulary.

    Returns:
        a pair: the vocabulary( a dictionary mapping string to integers), and
        the reversed vocabulary( a list, which reverse the vocabulary mapping).

    Raises:
            ValueError: if the provided vocabulary_path does not exist.
    """
    if tf.gfile.Exists(vocabulary_path):
        rev_vocab = []
        with tf.gfile.GFile(vocabulary_path, mode="r") as f:
            rev_vocab.extend(f.readlines())
        rev_vocab = [line.strip() for line in rev_vocab]
        vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
        # print(vocab)
        return vocab, rev_vocab
    else:
        raise ValueError("Vocabulary file %s not found." % vocabulary_path)


def sentence_to_token_ids(sentence, vocabulary, UNK_ID,
                          tokenizer=None, normalize_digits=False):
    if tokenizer:
        words = tokenizer(sentence)
    else:
        words = basic_tokenizer(sentence)
    if not normalize_digits:
        return [vocabulary.get(w, UNK_ID) for w in words]
    # Normalize digits by 0 before looking words up in the vocabulary.
    return [vocabulary.get(re.sub(_DIGIT_RE, "0", w), UNK_ID) for w in words]


def data_to_token_ids(data_path, target_path, vocabulary_path,
                      tokenizer=None, normalize_digits=False, use_padding=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data-path, calls the above
    sentence_to_token_ids, and saves the result to target_path.See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        data_path: path to the data file in one-sentence-per-line format.
        target_path: path where the file with token-ids will be created.
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
    """
    if not tf.gfile.Exists(target_path):
        print("Tokenizing data in %s" % data_path)
        vocab, _ = initialize_vocabulary(vocabulary_path)
        with tf.gfile.GFile(data_path, mode="r") as data_file:
            with tf.gfile.GFile(target_path, mode="w") as tokens_file:
                counter = 0
                for line in data_file:
                    counter += 1
                    if counter % 100000 == 0:
                        print(" tokenizing line %d" % counter)
                    if use_padding:
                        UNK_ID = UNK_ID_dict["with_padding"]
                        token_ids = sentence_to_token_ids(line, vocab, UNK_ID, tokenizer,
                                                          normalize_digits)
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")
                    else:
                        words = line.strip().split()
                        # print(words)
                        token_ids = [vocab.get(w) for w in words]
                        tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")


def nlu_input_to_token_ids(nlu_inputs, vocabulary_path, tokenizer=None, normalize_digits=False, use_padding=True):
    """Tokenize data file and turn into token-ids using given vocabulary file.

    This function loads data line-by-line from data-path, calls the above
    sentence_to_token_ids, and saves the result to target_path.See comment
    for sentence_to_token_ids on the details of token-ids format.

    Args:
        nlu_inputs: a list of words
        vocabulary_path: path to the vocabulary file.
        tokenizer: a function to use to tokenize each sentence;
            if None, basic_tokenizer will be used.
        normalize_digits: Boolean; if true, all digits are replaced by 0s.
        use_padding:
    """
    inputs = ' '.join(nlu_inputs)
    print("Tokenizing data in %s" % inputs)
    vocab, _ = initialize_vocabulary(vocabulary_path)

    if use_padding:
        UNK_ID = UNK_ID_dict["with_padding"]
    else:
        UNK_ID = UNK_ID_dict["no_padding"]
    token_ids = sentence_to_token_ids(inputs, vocab, UNK_ID, tokenizer,
                                      normalize_digits)

    return token_ids


def create_label_vocab(vocabulary_path, data_path):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="r") as f:
            counter = 0
            slot_counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                label = line.strip()
                if label != 'NULL':
                    slot_counter += 1
                vocab[label] = 1
            label_list = sorted(vocab)
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for k in label_list:
                    # print(k)
                    vocab_file.write(k + "\n")
            slot = data_path.split('/')[-1].split('.')[0].split('_')[-1]
            print('%s appears %d times in training data.' % (slot, slot_counter))


def create_intent_vocab(vocabulary_path, data_path):
    if not tf.gfile.Exists(vocabulary_path):
        print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
        vocab = {}
        with tf.gfile.GFile(data_path, mode="r") as f:
            counter = 0
            confirm_counter = 0
            inform_counter = 0
            reject_counter = 0
            for line in f:
                counter += 1
                if counter % 100000 == 0:
                    print(" processing line %d" % counter)
                label = line.strip()
                if label == 'confirm':
                    confirm_counter += 1
                elif label == 'inform':
                    inform_counter += 1
                elif label == 'reject':
                    reject_counter += 1
                vocab[label] = 1
            label_list = sorted(vocab)
            with tf.gfile.GFile(vocabulary_path, mode="w") as vocab_file:
                for k in label_list:
                    # print(k)
                    vocab_file.write(k + "\n")
            # slot = data_path.split('/')[-1].split('.')[0]
            print('inform intent appears %d times in training data.' % inform_counter)
            print('confirm intent appears %d times in training data.' % confirm_counter)
            print('reject intent appears %d times in training data.' % reject_counter)


def prepare_multi_task_data(data_dir, sent_vocab_size):
    train_path = data_dir + "/train/train_normal"
    valid_path = data_dir + "/valid/valid_normal"
    test_path = data_dir + "/test/test_normal"

    # Create vocabularies of the appropriate sizes.
    sent_vocab_path = os.path.join(data_dir, "sent_vocab_%d.txt" % sent_vocab_size)
    s_attr_vocab_path = os.path.join(data_dir, "s_attr.txt")
    s_loc_vocab_path = os.path.join(data_dir, "s_loc.txt")
    s_name_vocab_path = os.path.join(data_dir, "s_name.txt")
    s_ope_vocab_path = os.path.join(data_dir, "s_ope.txt")
    intent_vocab_path = os.path.join(data_dir, "intent.txt")
    slot_vocab_path = [s_attr_vocab_path,
                       s_loc_vocab_path,
                       s_name_vocab_path,
                       s_ope_vocab_path]

    create_vocabulary(sent_vocab_path, train_path + "_sent.txt", sent_vocab_size, tokenizer=naive_tokenizer)
    create_label_vocab(s_attr_vocab_path, train_path + "_s_attr.txt")
    create_label_vocab(s_loc_vocab_path, train_path + "_s_loc.txt")
    create_label_vocab(s_name_vocab_path, train_path + "_s_name.txt")
    create_label_vocab(s_ope_vocab_path, train_path + "_s_ope.txt")
    create_intent_vocab(intent_vocab_path, train_path + "_intent.txt")

    # Create token ids for the training data.
    sent_train_ids_path = train_path + ("_ids%d_sent.txt" % sent_vocab_size)
    s_attr_train_ids_path = train_path + "_ids_s_attr.txt"
    s_loc_train_ids_path = train_path + "_ids_s_loc.txt"
    s_name_train_ids_path = train_path + "_ids_s_name.txt"
    s_ope_train_ids_path = train_path + "_ids_s_ope.txt"
    intent_train_ids_path = train_path + "_ids_intent.txt"
    slot_train_ids_path = [s_attr_train_ids_path, 
                           s_loc_train_ids_path, 
                           s_name_train_ids_path, 
                           s_ope_train_ids_path]

    data_to_token_ids(train_path + "_sent.txt", sent_train_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(train_path + "_s_attr.txt", s_attr_train_ids_path, s_attr_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(train_path + "_s_loc.txt", s_loc_train_ids_path, s_loc_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(train_path + "_s_name.txt", s_name_train_ids_path, s_name_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(train_path + "_s_ope.txt", s_ope_train_ids_path, s_ope_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(train_path + "_intent.txt", intent_train_ids_path, intent_vocab_path, normalize_digits=False,
                      use_padding=False)

    # Create token ids for the development data.
    sent_valid_ids_path = valid_path + ("_ids%d_sent.txt" % sent_vocab_size)
    s_attr_valid_ids_path = valid_path + "_ids_s_attr.txt"
    s_loc_valid_ids_path = valid_path + "_ids_s_loc.txt"
    s_name_valid_ids_path = valid_path + "_ids_s_name.txt"
    s_ope_valid_ids_path = valid_path + "_ids_s_ope.txt"
    intent_valid_ids_path = valid_path + "_ids_intent.txt"
    slot_valid_ids_path = [s_attr_valid_ids_path, 
                           s_loc_valid_ids_path, 
                           s_name_valid_ids_path, 
                           s_ope_valid_ids_path]

    data_to_token_ids(valid_path + "_sent.txt", sent_valid_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(valid_path + "_s_attr.txt", s_attr_valid_ids_path, s_attr_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(valid_path + "_s_loc.txt", s_loc_valid_ids_path, s_loc_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(valid_path + "_s_name.txt", s_name_valid_ids_path, s_name_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(valid_path + "_s_ope.txt", s_ope_valid_ids_path, s_ope_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(valid_path + "_intent.txt", intent_valid_ids_path, intent_vocab_path, normalize_digits=False,
                      use_padding=False)

    # Create token ids for the test data.
    sent_test_ids_path = test_path + ("_ids%d_sent.txt" % sent_vocab_size)
    s_attr_test_ids_path = test_path + "_ids_s_attr.txt"
    s_loc_test_ids_path = test_path + "_ids_s_loc.txt"
    s_name_test_ids_path = test_path + "_ids_s_name.txt"
    s_ope_test_ids_path = test_path + "_ids_s_ope.txt"
    intent_test_ids_path = test_path + "_ids_intent.txt"
    slot_test_ids_path = [s_attr_test_ids_path,
                          s_loc_test_ids_path,
                          s_name_test_ids_path,
                          s_ope_test_ids_path]

    data_to_token_ids(test_path + "_sent.txt", sent_test_ids_path, sent_vocab_path, tokenizer=naive_tokenizer)
    data_to_token_ids(test_path + "_s_attr.txt", s_attr_test_ids_path, s_attr_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(test_path + "_s_loc.txt", s_loc_test_ids_path, s_loc_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(test_path + "_s_name.txt", s_name_test_ids_path, s_name_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(test_path + "_s_ope.txt", s_ope_test_ids_path, s_ope_vocab_path, normalize_digits=False,
                      use_padding=False)
    data_to_token_ids(test_path + "_intent.txt", intent_test_ids_path, intent_vocab_path, normalize_digits=False,
                      use_padding=False)

    return (sent_train_ids_path, slot_train_ids_path, intent_train_ids_path,
            sent_valid_ids_path, slot_valid_ids_path, intent_valid_ids_path,
            sent_test_ids_path, slot_test_ids_path, intent_test_ids_path,
            sent_vocab_path, slot_vocab_path, intent_vocab_path)


def read_data(sent_path, slot_path, intent_path, max_size=None):
    """Read data from source and target files and put into buckets.

  Args:
    sent_path: path to the files with token-ids for the source input - word sequence.
    slot_path: path to the file with token-ids for the target output - tag sequence;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    intent_path: path to the file with token-ids for the sequence classification label
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target, label) tuple read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source,  target, and label are lists of token-ids.
  """
    data_set = []  # len(data_set) always 1
    with tf.gfile.GFile(sent_path, mode="r") as sent_file:
        with tf.gfile.GFile(slot_path[0], mode="r") as s_attr_file:
            with tf.gfile.GFile(slot_path[1], mode="r") as s_loc_file:
                with tf.gfile.GFile(slot_path[2], mode="r") as s_name_file:
                    with tf.gfile.GFile(slot_path[3], mode="r") as s_ope_file:
                        with tf.gfile.GFile(intent_path, mode="r") as intent_file:
                            sent = sent_file.readline()
                            s_attr = s_attr_file.readline().strip()
                            s_loc = s_loc_file.readline().strip()
                            s_name = s_name_file.readline().strip()
                            s_ope = s_ope_file.readline().strip()
                            intent = intent_file.readline().strip()
                            counter = 0
                            while sent and s_attr and s_loc and s_name and s_ope and intent and (
                                        not max_size or counter < max_size):
                                counter += 1
                                if counter % 100000 == 0:
                                    print("  reading data line %d" % counter)
                                    sys.stdout.flush()
                                sent_ids = [int(x) for x in sent.split()]
                                s_attr_ids = [int(s_attr)]
                                s_loc_ids = [int(s_loc)]
                                s_name_ids = [int(s_name)]
                                s_ope_ids = [int(s_ope)]
                                intent_ids = [int(intent)]
                                data_set.append(
                                    [sent_ids, s_attr_ids, s_loc_ids, s_name_ids, s_ope_ids, intent_ids])

                                sent = sent_file.readline()
                                s_attr = s_attr_file.readline()
                                s_loc = s_loc_file.readline()
                                s_name = s_name_file.readline()
                                s_ope = s_ope_file.readline()
                                intent = intent_file.readline()
    # 7 outputs in each unit : sent_ids, s_attr_ids, s_loc_ids, s_name_ids, s_ope_ids, s_way_ids, intent_ids]
    return data_set


# def main():
#     prepare_multi_task_data('./data', 500)
#     print('Done prepare multi task data.')
#
#
# if __name__ == '__main__':
#     main()





















