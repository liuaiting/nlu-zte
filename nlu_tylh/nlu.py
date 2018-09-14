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

import multi_task_model
import data_utils


tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 32, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 100, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("sent_vocab_size", 500, "max vocab Size.")
tf.app.flags.DEFINE_integer("alpha", 0.5, "slot weight.")
tf.app.flags.DEFINE_string("data_dir", "./data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8,
                          "dropout keep cell input and output prob.")
FLAGS = tf.app.flags.FLAGS

sent_vocab_path = os.path.join(FLAGS.data_dir, "sent_vocab_%d.txt" % FLAGS.sent_vocab_size)
s_attr_vocab_path = os.path.join(FLAGS.data_dir, "s_attr.txt")
s_loc_vocab_path = os.path.join(FLAGS.data_dir, "s_loc.txt")
s_name_vocab_path = os.path.join(FLAGS.data_dir, "s_name.txt")
s_ope_vocab_path = os.path.join(FLAGS.data_dir, "s_ope.txt")
intent_vocab_path = os.path.join(FLAGS.data_dir, "intent.txt")
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

# rev_vocab = [rev_s_attr_vocab, rev_s_loc_vocab, rev_s_name_vocab, rev_s_ope_vocab, rev_s_way_vocab, rev_intent_vocab]

sent_vocab_size = len(sent_vocab)
slot_vocab_size = [len(s_attr_vocab), len(s_loc_vocab), len(s_name_vocab), len(s_ope_vocab)]
intent_vocab_size = len(intent_vocab)


def load_nlu_model():
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    with tf.variable_scope("model", reuse=None):
        model_test = multi_task_model.MultiTaskModel(
            sent_vocab_size, slot_vocab_size, intent_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate, alpha=FLAGS.alpha,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=True)
    model_test.saver.restore(session, tf.train.get_checkpoint_state(FLAGS.train_dir).model_checkpoint_path)

    return session, model_test


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

    token_inputs = data_utils.nlu_input_to_token_ids(inputs, sent_vocab_path, data_utils.naive_tokenizer)

    data = [[token_inputs, [0], [0], [0], [0], [0]]]
    # print(data)

    np_inputs, s_attrs, s_locs, s_names, s_opes, intents, sequence_length = \
        model.get_one(data, 0)
    # print(np_inputs)

    _, _step_loss, logits = model.step(sess, np_inputs, s_attrs, s_locs, s_names,
                                       s_opes, intents, sequence_length, True)
    hyp_s_attr = rev_s_attr_vocab[np.argmax(logits[0])]
    hyp_s_loc = rev_s_loc_vocab[np.argmax(logits[1])]
    hyp_s_name = rev_s_name_vocab[np.argmax(logits[2])]
    hyp_s_ope = rev_s_ope_vocab[np.argmax(logits[3])]
    hyp_intent = rev_intent_vocab[np.argmax(logits[4])]

    # write prediction result to output file path
    # with tf.gfile.GFile(nlu_result_file, 'w') as f:
    #     f.write("s_attr: " + str(hyp_s_attr) + '\n')
    #     f.write("s_loc: " + str(hyp_s_loc) + '\n')
    #     f.write("s_name: " + str(hyp_s_name) + '\n')
    #     f.write("s_ope: " + str(hyp_s_ope) + '\n')
    #     f.write("s_way: " + str(hyp_s_way) + '\n')
    #     f.write("intent: " + str(hyp_intent) + '\n')
    # print type(hyp_s_attr)

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
sess, model = load_nlu_model()
test = json.dumps({"request_data": {"input": ['不', '是', '，', '关', '于', '初', '始', '密', '码', '。']},
                   "dr_result": {"domain": '翼支付'}})

result = nlu_interface(test, sess, model)

