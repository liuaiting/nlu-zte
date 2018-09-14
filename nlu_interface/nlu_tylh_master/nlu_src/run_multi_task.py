# -*- coding: utf-8 -*-
"""
Created on Thur Mar 2 2017

@author: Aiting Liu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import time

import numpy as np
import tensorflow as tf

from nlu_tylh_master.nlu_src import data_utils
from nlu_tylh_master.nlu_src import multi_task_model

import subprocess

tf.app.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
# tf.app.flags.DEFINE_float("learning_rate_decay_factor", 0.9,
#                          "Learning rate decays by this much.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0,
                          "Clip gradients to this norm.")
tf.app.flags.DEFINE_integer("batch_size", 8,
                            "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 32, "Size of each model layer.")
tf.app.flags.DEFINE_integer("word_embedding_size", 100, "Size of the word embedding")
tf.app.flags.DEFINE_integer("num_layers", 2, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("sent_vocab_size", 500, "max vocab Size.")
# tf.app.flags.DEFINE_integer("out_vocab_size", 500, "max tag vocab Size.")
tf.app.flags.DEFINE_integer("alpha", 0.5, "slot weight.")
tf.app.flags.DEFINE_string("data_dir", "../nlu_data", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "../nlu_model", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0,
                            "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 1000,
                            "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("max_training_steps", 1000000,
                            "Max training steps.")
tf.app.flags.DEFINE_integer("max_test_data_size", 0,
                            "Max size of test set.")
# tf.app.flags.DEFINE_boolean("use_attention", False,
#                             "Use attention based RNN")
tf.app.flags.DEFINE_integer("max_sequence_length", 50,
                            "Max sequence length.")
tf.app.flags.DEFINE_float("dropout_keep_prob", 0.8,
                          "dropout keep cell input and output prob.")
# tf.app.flags.DEFINE_boolean("bidirectional_rnn", False,
#                             "Use birectional RNN")
# tf.app.flags.DEFINE_string("task", 'joint', "Options: joint; intent; tagging")
FLAGS = tf.app.flags.FLAGS

if FLAGS.max_sequence_length == 0:
    print('Please indicate max sequence length. Exit')
    exit()


# if FLAGS.task is None:
#     print('Please indicate task to run. Available options: intent; tagging; joint')
#     exit()

# task = {'intent': 0, 'tagging': 0, 'joint': 0}
# if FLAGS.task == 'intent':
#     task['intent'] = 1
# elif FLAGS.task == 'tagging':
#     task['tagging'] = 1
# elif FLAGS.task == 'joint':
#     task['intent'] = 1
#     task['tagging'] = 1
#     task['joint'] = 1

# _buckets = [(FLAGS.max_sequence_length, FLAGS.max_sequence_length)]


def create_model(session, sent_vocab_size, slot_vocab_size, intent_vocab_size):
    """Create model and initialize or load parameters in session."""
    with tf.variable_scope("model_tylh", reuse=None):
        model_train = multi_task_model.MultiTaskModel(
            sent_vocab_size, slot_vocab_size, intent_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate, alpha=FLAGS.alpha,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=False)
        # use_attention=FLAGS.use_attention,
        # bidirectional_rnn=FLAGS.bidirectional_rnn,
        # task=task)
    with tf.variable_scope("model_tylh", reuse=True):
        model_test = multi_task_model.MultiTaskModel(
            sent_vocab_size, slot_vocab_size, intent_vocab_size, FLAGS.max_sequence_length,
            FLAGS.word_embedding_size, FLAGS.size, FLAGS.num_layers, FLAGS.max_gradient_norm, FLAGS.batch_size,
            learning_rate=FLAGS.learning_rate, alpha=FLAGS.alpha,
            dropout_keep_prob=FLAGS.dropout_keep_prob, use_lstm=True,
            forward_only=True)
        # use_attention=FLAGS.use_attention,
        # bidirectional_rnn=FLAGS.bidirectional_rnn,
        # task=task)

    ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
    if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model_train.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.global_variables_initializer())
    return model_train, model_test


def train():
    print('Applying Parameters:')
    for k, v in FLAGS.__dict__['__flags'].items():
        print('%s: %s' % (k, str(v)))
    print("Preparing data in %s" % FLAGS.data_dir)
    # sent_vocab_path = ''
    # s_attr_vocab_path = ''
    # s_loc_vocab_path = ''
    # s_name_vocab_path = ''
    # s_ope_vocab_path = ''
    # s_way_vocab_path = ''
    # intent_vocab_path = ''
    sent_train, slot_train, intent_train, \
    sent_valid, slot_valid, intent_valid, \
    sent_test, slot_test, intent_test, \
    sent_vocab_path, slot_vocab_path, intent_vocab_path = data_utils.prepare_multi_task_data(
        FLAGS.data_dir, FLAGS.sent_vocab_size)

    result_dir = FLAGS.data_dir + '/test_results'
    if not os.path.isdir(result_dir):
        os.makedirs(result_dir)

    current_valid_out_file = result_dir + '/valid_hyp'
    current_test_out_file = result_dir + '/test_hyp'
    current_train_out_file = result_dir + '/train_hyp'

    sent_vocab, rev_sent_vocab = data_utils.initialize_vocabulary(sent_vocab_path)
    s_attr_vocab, rev_s_attr_vocab = data_utils.initialize_vocabulary(slot_vocab_path[0])
    s_loc_vocab, rev_s_loc_vocab = data_utils.initialize_vocabulary(slot_vocab_path[1])
    s_name_vocab, rev_s_name_vocab = data_utils.initialize_vocabulary(slot_vocab_path[2])
    s_ope_vocab, rev_s_ope_vocab = data_utils.initialize_vocabulary(slot_vocab_path[3])
    intent_vocab, rev_intent_vocab = data_utils.initialize_vocabulary(intent_vocab_path)
    print(rev_intent_vocab)

    sent_vocab_size = len(sent_vocab)
    slot_vocab_size = [len(s_attr_vocab), len(s_loc_vocab), len(s_name_vocab), len(s_ope_vocab)]
    intent_vocab_size = len(intent_vocab)

    # print(sent_vocab_size, slot_vocab_size, intent_vocab_size)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # Create model.
        print("Max sequence length: %d." % FLAGS.max_sequence_length)
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))

        sess.run(tf.global_variables_initializer())

        model, model_test = create_model(sess, sent_vocab_size, slot_vocab_size, intent_vocab_size)
        print("Creating model with sent_vocab_size=%d, s_attr_vocab_size=%d, "
              "s_loc_vocab_size=%d, s_name_vocab_size=%d, "
              "s_ope_vocab_size=%d, and intent_vocab_size=%d." % (sent_vocab_size, slot_vocab_size[0],
                                                                  slot_vocab_size[1], slot_vocab_size[2],
                                                                  slot_vocab_size[3], intent_vocab_size))

        # Read data into buckets and compute their sizes.
        print("Reading train/valid/test data (training set limit: %d)."
              % FLAGS.max_train_data_size)
        valid_set = data_utils.read_data(sent_valid, slot_valid, intent_valid)
        test_set = data_utils.read_data(sent_test, slot_test, intent_test)
        train_set = data_utils.read_data(sent_train, slot_train, intent_train)

        # This is the training loop.
        step_time, loss = 0.0, 0.0
        current_step = 0

        best_valid_score = 0
        best_test_score = 0
        best_train_score = 0

        while model.global_step.eval() < FLAGS.max_training_steps:
            # Get a batch and make a step.
            start_time = time.time()

            batch_inputs, batch_s_attrs, batch_s_locs, batch_s_names, batch_s_opes, \
            batch_intents, batch_sequence_length = model.get_batch(train_set)
            # print(batch_inputs[0].shape)

            _, step_loss, logits = model.step(sess, batch_inputs, batch_s_attrs, batch_s_locs,
                                              batch_s_names, batch_s_opes, batch_intents,
                                              batch_sequence_length, False)
            # print(logits[-1])
            # print('s_attrs_logits', logits[0])
            # print(logits[0].shape)

            step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
            loss += step_loss / FLAGS.steps_per_checkpoint
            current_step += 1

            # Once in a while, we save checkpoint, print statistics, and run evals.
            if current_step % FLAGS.steps_per_checkpoint == 0:
                perplexity = math.exp(loss) if loss < 300 else float('inf')
                print("global step %d step-time %.2f. Training perplexity %.2f"
                      % (model.global_step.eval(), step_time, perplexity))
                sys.stdout.flush()
                # Save checkpoint and zero timer and loss.
                checkpoint_path = os.path.join(FLAGS.train_dir, "model.ckpt")
                model.saver.save(sess, checkpoint_path, global_step=model.global_step)
                step_time, loss = 0.0, 0.0
                #
                # def write_eval_result(result_list, result_path):
                #     with tf.gfile.GFile(result_path, 'w') as f:
                #         for i in range(len(result_list[0])):
                #             f.write(' '.join([str(result_list[j][i]) for j in range(5)]) + '\n')
                #
                # def run_valid_test(data_set, mode):  # mode: Eval, Test
                #     # Run evals on development/test set and print the accuracy.
                #     ref_s_attr_list = list()
                #     hyp_s_attr_list = list()
                #     ref_s_loc_list = list()
                #     hyp_s_loc_list = list()
                #     ref_s_name_list = list()
                #     hyp_s_name_list = list()
                #     ref_s_ope_list = list()
                #     hyp_s_ope_list = list()
                #     ref_intent_list = list()
                #     hyp_intent_list = list()
                #     s_attr_correct_count = 0
                #     s_loc_correct_count = 0
                #     s_name_correct_count = 0
                #     s_ope_correct_count = 0
                #     intent_correct_count = 0
                #
                #     # accuracy = 0.0
                #
                #     eval_loss = 0.0
                #     count = 0
                #     for i in range(len(data_set)):
                #         count += 1
                #         inputs, s_attrs, s_locs, s_names, s_opes, intents, sequence_length = \
                #             model_test.get_one(data_set, i)
                #         # print(s_attrs)
                #         # print(s_attrs.shape)
                #         # s_attr_logits = []
                #         # intents_logits = []
                #
                #         _, _step_loss, logits = model_test.step(sess, inputs, s_attrs, s_locs, s_names,
                #                                                 s_opes, intents, sequence_length, True)
                #         eval_loss += _step_loss / len(data_set)
                #         # hyp_s_attr = None
                #         # if task['intent'] == 1:
                #         # print(s_attrs[0][0])
                #         # print(logit)
                #         # print(logits[-1])
                #         ref_s_attr = np.argmax(s_attrs)
                #         ref_s_attr_list.append(rev_s_attr_vocab[ref_s_attr])
                #         hyp_s_attr = np.argmax(logits[0])
                #         hyp_s_attr_list.append(rev_s_attr_vocab[hyp_s_attr])
                #         # print(ref_s_attr, hyp_s_attr)
                #         ref_s_loc = np.argmax(s_locs)
                #         ref_s_loc_list.append(rev_s_loc_vocab[ref_s_loc])
                #         hyp_s_loc = np.argmax(logits[1])
                #         hyp_s_loc_list.append(rev_s_loc_vocab[hyp_s_loc])
                #         ref_s_name = np.argmax(s_names)
                #         ref_s_name_list.append(rev_s_name_vocab[ref_s_name])
                #         hyp_s_name = np.argmax(logits[2])
                #         hyp_s_name_list.append(rev_s_name_vocab[hyp_s_name])
                #         ref_s_ope = np.argmax(s_opes)
                #         ref_s_ope_list.append(rev_s_ope_vocab[ref_s_ope])
                #         hyp_s_ope = np.argmax(logits[3])
                #         hyp_s_ope_list.append(rev_s_ope_vocab[hyp_s_ope])
                #         ref_intent = np.argmax(intents)
                #         ref_intent_list.append(rev_intent_vocab[ref_intent])
                #         hyp_intent = np.argmax(logits[4])
                #         hyp_intent_list.append(rev_intent_vocab[hyp_intent])
                #         # print(hyp_intent, ref_intent)
                #         # print('s_attrs', s_attrs[0][0])
                #         # print('s_attr_logits', s_attr_logits[0])
                #         # print(ref_s_attr, hyp_s_attr)
                #         if ref_s_attr == hyp_s_attr:
                #             s_attr_correct_count += 1
                #         if ref_s_loc == hyp_s_loc:
                #             s_loc_correct_count += 1
                #         if ref_s_name == hyp_s_name:
                #             s_name_correct_count += 1
                #         if ref_s_ope == hyp_s_ope:
                #             s_ope_correct_count += 1
                #         if ref_intent == hyp_intent:
                #             intent_correct_count += 1
                #
                #     s_attr_accuracy = float(s_attr_correct_count) * 100 / count
                #     s_loc_accuracy = float(s_loc_correct_count) * 100 / count
                #     s_name_accuracy = float(s_name_correct_count) * 100 / count
                #     s_ope_accuracy = float(s_ope_correct_count) * 100 / count
                #     slot_accuracy = (s_attr_accuracy + s_loc_accuracy + s_name_accuracy
                #                      + s_ope_accuracy) / 4
                #
                #     intent_accuracy = float(intent_correct_count) * 100 / count
                #
                #     # if task['intent'] == 1:
                #     print("  %s s_attr_accuracy: %.2f %d/%d" % (mode, s_attr_accuracy, s_attr_correct_count, count))
                #     print("  %s s_loc_accuracy: %.2f %d/%d" % (mode, s_loc_accuracy, s_loc_correct_count, count))
                #     print("  %s s_name_accuracy: %.2f %d/%d" % (mode, s_name_accuracy, s_name_correct_count, count))
                #     print("  %s s_ope_accuracy: %.2f %d/%d" % (mode, s_ope_accuracy, s_ope_correct_count, count))
                #     print("  %s intent_accuracy: %.2f %d/%d" % (mode, intent_accuracy, intent_correct_count, count))
                #     sys.stdout.flush()
                #     out_file = None
                #     if mode == 'Eval':
                #         out_file = current_valid_out_file
                #     elif mode == 'Test':
                #         out_file = current_test_out_file
                #     elif mode == 'Train':
                #         out_file = current_train_out_file
                #
                #     hyp_list = [hyp_s_attr_list, hyp_s_loc_list, hyp_s_name_list, hyp_s_ope_list, hyp_intent_list]
                #     ref_list = [ref_s_attr_list, ref_s_loc_list, ref_s_name_list, ref_s_ope_list, ref_intent_list]
                #
                #     write_eval_result(hyp_list, out_file)  # write prediction result to output file path
                #
                #     return slot_accuracy, intent_accuracy, hyp_list
                #
                # # # train
                # # train_slot_accuracy, train_intent_accuracy, hyp_list = run_valid_test(train_set, 'Train')
                # # if train_slot_accuracy > best_train_score:
                # #     best_train_score = train_slot_accuracy
                # #     # save the best output file
                # #     subprocess.call(['mv', current_train_out_file,
                # #                      current_train_out_file + '_best_acc_%.2f' % best_train_score])
                #
                # # valid
                # valid_slot_accuracy, valid_intent_accuracy, hyp_list = run_valid_test(valid_set, 'Eval')
                # if valid_slot_accuracy > best_valid_score:
                #     best_valid_score = valid_slot_accuracy
                #     # save the best output file
                #     subprocess.call(['mv', current_valid_out_file,
                #                      current_valid_out_file + '_best_acc_%.2f' % best_valid_score])
                # # test, run test after each validation for development purpose.
                # test_slot_accuracy, test_intent_accuracy, hyp_list = run_valid_test(test_set, 'Test')
                # if test_slot_accuracy > best_test_score:
                #     best_test_score = test_slot_accuracy
                #     # save the best output file
                #     subprocess.call(['mv', current_test_out_file,
                #                      current_test_out_file + '_best_acc_%.2f' % best_test_score])
                #

def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()
