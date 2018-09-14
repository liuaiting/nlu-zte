# -*- coding: utf-8 -*-
"""
Created on Tue March 7 2017

@author: Aiting Liu

NLU :
    input: string, input sequence of words
    output: slot value , intent value
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
import json
import tensorflow as tf

import nlu_tylh_master.nlu_src.multi_task_model as multi_task_model
import nlu_tylh_master.nlu_src.data_utils as data_utils


class Config(object):

    learning_rate = 0.05
    max_gradient_norm = 5.0
    batch_size = 8,
    size = 32
    word_embedding_size = 100
    num_layers = 2
    sent_vocab_size = 500
    alpha = 0.5
    data_dir = "nlu_tylh_master/nlu_data"
    train_dir = "nlu_tylh_master/nlu_model"
    # data_dir = "../nlu_data"
    # train_dir = "../nlu_model"
    max_train_data_size = 0
    max_sequence_length = 50
    dropout_keep_prob = 0.8

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

    sent_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sent_vocab_path)
    s_attr_vocab, rev_s_attr_vocab = data_utils.initialize_vocabulary(slot_vocab_path[0])
    s_loc_vocab, rev_s_loc_vocab = data_utils.initialize_vocabulary(slot_vocab_path[1])
    s_name_vocab, rev_s_name_vocab = data_utils.initialize_vocabulary(slot_vocab_path[2])
    s_ope_vocab, rev_s_ope_vocab = data_utils.initialize_vocabulary(slot_vocab_path[3])
    intent_vocab, rev_intent_vocab = data_utils.initialize_vocabulary(intent_vocab_path)

    sent_vocab_size = len(sent_vocab)
    slot_vocab_size = [len(s_attr_vocab), len(s_loc_vocab), len(s_name_vocab), len(s_ope_vocab)]
    intent_vocab_size = len(intent_vocab)


config = Config()


def load_nlu_model(sess):
    # sess.run(tf.global_variables_initializer())
    with tf.variable_scope("model_tylh", reuse=None):
        model_test = multi_task_model.MultiTaskModel(
            config.sent_vocab_size, config.slot_vocab_size, config.intent_vocab_size, config.max_sequence_length,
            config.word_embedding_size, config.size, config.num_layers, config.max_gradient_norm, config.batch_size,
            learning_rate=config.learning_rate, alpha=config.alpha,
            dropout_keep_prob=config.dropout_keep_prob, use_lstm=True,
            forward_only=True)
    sess.run(tf.global_variables_initializer())
    ckpt = tf.train.get_checkpoint_state(config.train_dir)
    if ckpt:
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_test.saver.restore(sess, ckpt.model_checkpoint_path)
    return model_test


def nlu_interface(nlu_inputs, sess, model):
    """
    processing nlu, get slot filling and intent detection results.
    Args:
        nlu_inputs: json string, contain a list of words.
        sess: session
        model: model from latest checkpint
    Return:
        nlu_results: json string, slot filling result ang intent detection results.
    """
    assert type(nlu_inputs) == str
    # print("-"*30 + "NLU MODULE" + "-"*30)
    inputs = json.loads(nlu_inputs)["request_data"]["input"]
    domain = json.loads(nlu_inputs)["dr_result"]["domain"]
    # print("nlu input: %s" % json.loads(nlu_inputs))

    token_inputs = data_utils.nlu_input_to_token_ids(inputs, config.sent_vocab_path, data_utils.naive_tokenizer)

    data = [[token_inputs, [0], [0], [0], [0], [0]]]
    # print(data)

    np_inputs, s_attrs, s_locs, s_names, s_opes, intents, sequence_length = \
        model.get_one(data, 0)
    # print(np_inputs)

    _, _step_loss, logits = model.step(sess, np_inputs, s_attrs, s_locs, s_names,
                                       s_opes, intents, sequence_length, True)
    hyp_s_attr = config.rev_s_attr_vocab[np.argmax(logits[0])]
    hyp_s_loc = config.rev_s_loc_vocab[np.argmax(logits[1])]
    hyp_s_name = config.rev_s_name_vocab[np.argmax(logits[2])]
    hyp_s_ope = config.rev_s_ope_vocab[np.argmax(logits[3])]
    hyp_intent = config.rev_intent_vocab[np.argmax(logits[4])]

    nlu_output = {"nlu_result": {"intent": hyp_intent,
                                 "domain": domain,
                                 "slots": []}}

    if hyp_s_attr != "NULL":
        nlu_output["nlu_result"]["slots"].append({"slot_name": "attr",
                                                  "slot_val": hyp_s_attr,
                                                  "confidence": "1"})
        # print('attr:', hyp_s_attr)
    if hyp_s_loc != "NULL":
        nlu_output["nlu_result"]["slots"].append({"slot_name": "loc",
                                                  "slot_val": hyp_s_loc,
                                                  "confidence": "1"})
        # print("loc:", hyp_s_loc)
    if hyp_s_name != "NULL":
        nlu_output["nlu_result"]["slots"].append({"slot_name": "name",
                                                  "slot_val": hyp_s_name,
                                                  "confidence": "1"})
        # print("name:", hyp_s_name)
    if hyp_s_ope != "NULL":
        nlu_output["nlu_result"]["slots"].append({"slot_name": "ope",
                                                  "slot_val": hyp_s_ope,
                                                  "confidence": "1"})
        # print("ope:", hyp_s_ope)
    # print(nlu_output)

    # # print("nlu output: %s" % nlu_output)
    # # print("write nlu output in path: %s" % nlu_result_file)
    # with tf.gfile.GFile(nlu_result_file, 'w') as f:
    #     f.write(json.dumps(nlu_output))
    #
    # # print("-"*30 + "END NLU" + "-"*30)
    nlu_output = json.dumps(nlu_output)
    return nlu_output

# Example.
# sess = tf.Session()
#
# model = load_nlu_model(sess)
# test = json.dumps({"request_data": {"input": ['对', '于', '贺', '州', '用', '户', '来', '说', '，', '天','翼','领','航','A','8','共','享','版','光','纤','套','餐','如','何','办','理','？']},
#                    "dr_result": {"domain": '天翼领航'}})
#
# result = nlu_interface(test, sess, model)
# print(json.loads(result))


