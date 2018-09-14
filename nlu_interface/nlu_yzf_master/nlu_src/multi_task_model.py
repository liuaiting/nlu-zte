# -*- coding: utf-8 -*-
"""
Created on Thur Mar 2 2017

@author: Aiting Liu

Multi-task RNN model

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf

import nlu_yzf_master.nlu_src.data_utils as data_utils


class MultiTaskModel(object):
    """Wait for completing ......"""
    def __init__(self,
                 sent_vocab_size,
                 slot_vocab_size,
                 intent_vocab_size,
                 max_sequence_length,
                 word_embedding_size,
                 size,
                 num_layers,
                 max_gradient_norm,
                 batch_size,
                 learning_rate,
                 # learning_rate_decay_factor,
                 alpha=0.5,
                 dropout_keep_prob=0.8,
                 use_lstm=True,
                 # num_samples=512,
                 forward_only=False,
                 dtype=tf.float32):
        """Create the model.

        Args:
            sent_vocab_size: int, size of the source sentence vocabulary.
            slot_vocab_size: list, each size of a particular slot vocabulary.
            intent_vocab_size: int, size of the intent label vocabulary. dummy, only one intent.
            max_sequence_length: int, specifies maximum input length.
                Training instances' inputs will be padded accordingly.
            size: number of units in each layer of the model.
            num_layers: number of layers in the model.
            max_gradient_norm: gradients will be clipped to maximally this norm.
            batch_size: the size of the batches used during training;
                the model construction is independent of batch_size, so it can be
                changed after initialization if this is convenient, e.g., for decoding.
            learning_rate: learning rate to start with.
            # learning_rate_decay_factor: decay learning rate by this much when needed.
            use_lstm: if true, we use LSTM cells instead of GRU cells.
            # num_samples: number of samples for sampled softmax.
            forward_only: if set, we do not construct the backward pass in the model.
            dtype: the data type to use to store internal variables.
        """
        self.sent_vocab_size = sent_vocab_size
        self.slot_vocab_size = slot_vocab_size
        self.intent_vocab_size = intent_vocab_size
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.learning_rate = tf.Variable(
            float(learning_rate), trainable=False, dtype=dtype)
        # self.learning_rate_decay_op = self.learning_rate.assign(
        #     self.learning_rate * learning_rate_decay_factor)
        self.global_step = tf.Variable(0, trainable=False)

        # Feeds for inputs.
        self.sequence_length = tf.placeholder(tf.int32, [None], name="sequence_length")
        self.inputs = tf.placeholder(tf.int32, shape=[None, self.max_sequence_length], name="input")
        self.s_attrs = tf.placeholder(tf.float32, shape=[None, self.slot_vocab_size[0]], name="s_attr")
        self.s_locs = tf.placeholder(tf.float32, shape=[None, self.slot_vocab_size[1]], name="s_loc")
        self.s_names = tf.placeholder(tf.float32, shape=[None, self.slot_vocab_size[2]], name="s_name")
        self.s_opes = tf.placeholder(tf.float32, shape=[None, self.slot_vocab_size[3]], name="s_ope")
        self.s_ways = tf.placeholder(tf.float32, shape=[None, self.slot_vocab_size[4]], name="s_way")
        self.intents = tf.placeholder(tf.float32, shape=[None, self.intent_vocab_size], name="intent")

        # Create the internal multi-layer cell for our RNN.
        single_cell = tf.contrib.rnn.GRUCell(size)
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(num_units=size, state_is_tuple=True)
        cell = single_cell
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)

        if not forward_only and dropout_keep_prob < 1.0:
            cell = tf.contrib.rnn.DropoutWrapper(cell,
                                                 input_keep_prob=dropout_keep_prob,
                                                 output_keep_prob=dropout_keep_prob)

        # init_state = cell.zero_state(batch_size, tf.float32)

        embedding = tf.get_variable("embedding", [sent_vocab_size, word_embedding_size])
        self.embedded_inputs = tf.nn.embedding_lookup(embedding, self.inputs)

        # print(embedding.name)

        # Training outputs and final state.
        self.outputs, self.state = tf.nn.dynamic_rnn(
            cell, self.embedded_inputs, sequence_length=self.sequence_length, dtype=dtype)

        # get the last time step output.
        # output = tf.transpose(self.outputs, [1, 0, 2])
        # self.last = tf.gather(output, int(output.get_shape()[0]) - 1)
        self.slot_last = self.state[-1].h
        self.intent_last = self.state[0].h
        assert self.slot_last != self.intent_last
        # self.intent_last = self.state[-1].h

        # print(self.last)  # shape batch_size, hidden_size

        # Output projection.
        s_attr_weight = tf.get_variable('s_attr_weight', shape=[size, slot_vocab_size[0]], dtype=dtype, trainable=True)
        s_attr_bias = tf.get_variable('s_attr_bias', shape=[slot_vocab_size[0]], dtype=dtype)
        s_loc_weight = tf.get_variable('s_loc_weight', shape=[size, slot_vocab_size[1]], dtype=dtype)
        s_loc_bias = tf.get_variable('s_loc_bias', shape=[slot_vocab_size[1]], dtype=dtype)
        s_name_weight = tf.get_variable('s_name_weight', shape=[size, slot_vocab_size[2]], dtype=dtype)
        s_name_bias = tf.get_variable('s_name_bias', shape=[slot_vocab_size[2]], dtype=dtype)
        s_ope_weight = tf.get_variable('s_ope_weight', shape=[size, slot_vocab_size[3]], dtype=dtype)
        s_ope_bias = tf.get_variable('s_ope_bias', shape=[slot_vocab_size[3]], dtype=dtype)
        s_way_weight = tf.get_variable('s_way_weight', shape=[size, slot_vocab_size[4]], dtype=dtype)
        s_way_bias = tf.get_variable('s_way_bias', shape=[slot_vocab_size[4]], dtype=dtype)
        intent_weight = tf.get_variable('intent_weight', shape=[size, intent_vocab_size], dtype=dtype)
        intent_bias = tf.get_variable('intent_bias', shape=[intent_vocab_size], dtype=dtype)

        # print(s_attr_weight)

        # Training outputs.
        self.s_attr_outputs = tf.nn.xw_plus_b(self.slot_last, s_attr_weight, s_attr_bias)
        self.s_loc_outputs = tf.nn.xw_plus_b(self.slot_last, s_loc_weight, s_loc_bias)
        self.s_name_outputs = tf.nn.xw_plus_b(self.slot_last, s_name_weight, s_name_bias)
        self.s_ope_outputs = tf.nn.xw_plus_b(self.slot_last, s_ope_weight, s_ope_bias)
        self.s_way_outputs = tf.nn.xw_plus_b(self.slot_last, s_way_weight, s_way_bias)
        self.intent_outputs = tf.nn.xw_plus_b(self.intent_last, intent_weight, intent_bias)

        # Training logits.
        self.s_attr_logits = tf.nn.softmax(self.s_attr_outputs)
        self.s_loc_logits = tf.nn.softmax(self.s_loc_outputs)
        self.s_name_logits = tf.nn.softmax(self.s_name_outputs)
        self.s_ope_logits = tf.nn.softmax(self.s_ope_outputs)
        self.s_way_logits = tf.nn.softmax(self.s_way_outputs)
        self.intent_logits = tf.nn.softmax(self.intent_outputs)

        self.slot_logits = [self.s_attr_logits, self.s_loc_logits, self.s_name_logits,
                            self.s_ope_logits, self.s_way_logits]

        # Training cross entropy.
        self.s_attr_crossent = -tf.reduce_sum(self.s_attrs * tf.log(self.s_attr_logits))
        self.s_loc_crossent = -tf.reduce_sum(self.s_locs * tf.log(self.s_loc_logits))
        self.s_name_crossent = -tf.reduce_sum(self.s_names * tf.log(self.s_name_logits))
        self.s_ope_crossent = -tf.reduce_sum(self.s_opes * tf.log(self.s_ope_logits))
        self.s_way_crossent = -tf.reduce_sum(self.s_ways * tf.log(self.s_way_logits))
        self.intent_crossent = -tf.reduce_sum(self.intents * tf.log(self.intent_logits))
        # self.s_attr_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_attr_outputs, labels=self.s_attrs)
        # self.s_loc_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_loc_outputs, labels=self.s_locs)
        # self.s_name_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_name_outputs, labels=self.s_names)
        # self.s_ope_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_ope_outputs, labels=self.s_opes)
        # self.s_way_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_way_outputs, labels=self.s_ways)
        # self.intent_crossent = tf.nn.softmax_cross_entropy_with_logits(logits=self.intent_outputs, labels=self.intents)

        # Training loss.
        self.s_attr_loss = tf.reduce_sum(self.s_attr_crossent) / tf.cast(batch_size, tf.float32)
        self.s_loc_loss = tf.reduce_sum(self.s_loc_crossent) / tf.cast(batch_size, tf.float32)
        self.s_name_loss = tf.reduce_sum(self.s_name_crossent) / tf.cast(batch_size, tf.float32)
        self.s_ope_loss = tf.reduce_sum(self.s_ope_crossent) / tf.cast(batch_size, tf.float32)
        self.s_way_loss = tf.reduce_sum(self.s_way_crossent) / tf.cast(batch_size, tf.float32)
        self.slot_loss = (self.s_attr_loss + self.s_loc_loss + self.s_name_loss + self.s_ope_loss +
                          self.s_way_loss) / 5

        self.intent_loss = tf.reduce_sum(self.intent_crossent) / tf.cast(batch_size, tf.float32)

        self.losses = alpha * self.slot_loss + (1.0 - alpha) * self.intent_loss
        # self.losses = [self.slot_loss, self.intent_loss]

        # tf.summary.scalar('slot_loss', self.slot_loss)
        # tf.summary.scalar('intent_loss', self.intent_loss)
        # tf.summary.scalar('loss', self.losses)

        # Gradients and SGD update operation for training the model.
        params = tf.trainable_variables()
        if not forward_only:
            opt = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
            gradients = tf.gradients(self.losses, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                             max_gradient_norm)
            self.gradient_norm = norm
            self.update = opt.apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables())

    # @property
    # def input(self):
    #     return self.inputs
    #
    # @property
    # def output(self):
    #     return self.outputs
    #
    # @property
    # def slot_logit(self):
    #     return self.slot_logits
    #
    # @property
    # def intent_logit(self):
    #     return self.intent_logits

    def step(self, session, inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents,
             batch_sequence_length, forward_only):
        """Run a step of the model feeding the given inputs.

        Args:
            session: tensorflow session to use.
            inputs: list of numpy int vectors to feed as encoder inputs.
            s_attrs: numpy float vectors to feed as target s_attr label with shape=[batch_size, s_attr_vocab_size].
            s_locs: numpy float vectors to feed as target s_loc label with shape=[batch_size, s_loc_vocab_size].
            s_names: numpy float vectors to feed as target s_name label with shape=[batch_size, s_name_vocab_size].
            s_opes: numpy float vectors to feed as target s_ope label with shape=[batch_size, s_ope_vocab_size].
            s_ways: numpy float vectors to feed as target s_way label with shape=[batch_size, s_way_vocab_size].
            intents: numpy float vectors to feed as target intent label with shape=[batch_size, intent_vocab_size].
            batch_sequence_length: numpy float vectors to feed as sequence real length with shape=[batch_size, ].
            forward_only: whether to do the backward step or only forward.

        Returns:
            A triple consisting of gradient norm (or None if we did not do backward),
            average perplexity, and the outputs.

        Raises:
            ValueError: if length/shape of inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents, disagrees
            with the expected length/shape.
        """
        # Check if the sizes match.
        input_size = self.max_sequence_length
        # if len(inputs) != input_size:
        #     raise ValueError("Inputs length must be equal to the config max sequence length,"
        #                      " %d != %d." % (len(inputs), input_size))
        # if s_attrs.shape != (self.batch_size, self.slot_vocab_size[0]):
        #     raise ValueError("s_attrs.shape must be equal to the expected shape.")

        # Input feed: inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents, sequence_length as provided.
        input_feed = dict()
        input_feed[self.sequence_length.name] = batch_sequence_length
        input_feed[self.inputs.name] = inputs
        input_feed[self.s_attrs.name] = s_attrs
        input_feed[self.s_locs.name] = s_locs
        input_feed[self.s_names.name] = s_names
        input_feed[self.s_opes.name] = s_opes
        input_feed[self.s_ways.name] = s_ways
        input_feed[self.intents.name] = intents

        # Output feed: depends on whether we do a backward step or not.
        if not forward_only:
            output_feed = [self.update,  # Update Op that does SGD.
                           self.gradient_norm,  # Gradient norm.
                           self.losses,  # Loss for this batch.
                           self.s_attr_logits, self.s_loc_logits,  # Output logits.
                           self.s_name_logits, self.s_ope_logits,
                           self.s_way_logits, self.intent_logits]

        else:
            output_feed = [self.losses,  # Loss for this batch.
                           self.s_attr_logits, self.s_loc_logits,  # Loss for this batch.
                           self.s_name_logits, self.s_ope_logits,
                           self.s_way_logits, self.intent_logits]

        outputs = session.run(output_feed, input_feed)
        if not forward_only:
            return outputs[1], outputs[2], outputs[3:]  # Gradient norm, loss, no outputs.
        else:
            return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

    def get_batch(self, data):
        """Get a random batch of data from the data, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            self: get some configure
            data: a list in which each element contains
                lists of pairs of input and output data that we use to create a batch.

        Returns:
          The triple (inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents,
          sequence_length) for the constructed batch that has the proper format
          to call step(...) later.
        """
        input_size = self.max_sequence_length
        s_attr_size = self.slot_vocab_size[0]
        s_loc_size = self.slot_vocab_size[1]
        s_name_size = self.slot_vocab_size[2]
        s_ope_size = self.slot_vocab_size[3]
        s_way_size = self.slot_vocab_size[4]
        intent_size = self.intent_vocab_size

        inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents = [], [], [], [], [], [], []
        batch_sequence_length_list = list()

        # Get a random batch of inputs, targets and labels from data,
        # pad them if needed.
        for _ in range(self.batch_size):
            _input, _s_attr, _s_loc, _s_name, _s_ope, _s_way, _intent = random.choice(data)
            # batch_sequence_length_list.append(len(_input))

            # Inputs are padded.
            if len(_input) > input_size:
                batch_sequence_length_list.append(input_size)
                inputs.append(list(_input[:input_size]))
            else:
                batch_sequence_length_list.append(len(_input))
                input_pad = [data_utils.PAD_ID] * (input_size - len(_input))
                inputs.append(list(_input + input_pad))

            # labels don't need padding.
            s_attrs.append(_s_attr)
            s_locs.append(_s_loc)
            s_names.append(_s_name)
            s_opes.append(_s_ope)
            s_ways.append(_s_way)
            intents.append(_intent)

        # Now we create batch-major vectors from the data selected above.
        batch_inputs = np.array(inputs, dtype=np.int32)

        def one_hot(vector, num_classes):
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0

            if num_classes is None:
                num_classes = np.max(vector) + 1
            else:
                assert num_classes > 0
                assert num_classes >= np.max(vector)

            result = np.zeros(shape=(len(vector), num_classes))
            result[np.arange(len(vector)), vector] = 1
            return result.astype(int)

        batch_s_attrs = one_hot(np.array([s_attrs[batch_idx][0] for batch_idx in range(self.batch_size)],
                                         dtype=np.int32), s_attr_size)
        batch_s_locs = one_hot(np.array([s_locs[batch_idx][0] for batch_idx in range(self.batch_size)],
                                        dtype=np.int32), s_loc_size)
        batch_s_names = one_hot(np.array([s_names[batch_idx][0] for batch_idx in range(self.batch_size)],
                                         dtype=np.int32), s_name_size)
        batch_s_opes = one_hot(np.array([s_opes[batch_idx][0] for batch_idx in range(self.batch_size)],
                                        dtype=np.int32), s_ope_size)
        batch_s_ways = one_hot(np.array([s_ways[batch_idx][0] for batch_idx in range(self.batch_size)],
                                        dtype=np.int32), s_way_size)
        batch_intents = one_hot(np.array([intents[batch_idx][0] for batch_idx in range(self.batch_size)],
                                         dtype=np.int32), intent_size)

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
        # print('batch_inputs', batch_inputs)
        # print('batch_s_attrs', batch_s_attrs)

        return batch_inputs, batch_s_attrs, batch_s_locs, batch_s_names, batch_s_opes, batch_s_ways, batch_intents, batch_sequence_length

    def get_one(self, data, sample_id):
        """Get a single sample data from data, prepare for step.

        To feed data in step(..) it must be a list of batch-major vectors, while
        data here contains single length-major cases. So the main logic of this
        function is to re-index data cases to be in the proper format for feeding.

        Args:
            self: get some configure.
            data: a list in which each element contains lists of pairs of input
                and output data that we use to create a batch.
            sample_id: integer, which sample to get the batch for.

        Returns:
            The tuple (inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents,
            sequence_length) for the constructed batch that has the proper format
             to call step(...) later.
        """
        input_size = self.max_sequence_length
        s_attr_size = self.slot_vocab_size[0]
        s_loc_size = self.slot_vocab_size[1]
        s_name_size = self.slot_vocab_size[2]
        s_ope_size = self.slot_vocab_size[3]
        s_way_size = self.slot_vocab_size[4]
        intent_size = self.intent_vocab_size

        inputs, s_attrs, s_locs, s_names, s_opes, s_ways, intents = [], [], [], [], [], [], []
        batch_sequence_length_list = list()

        # Get a random batch of inputs, targets and labels from data,
        # pad them if needed.
        _input, _s_attr, _s_loc, _s_name, _s_ope, _s_way, _intent = data[sample_id]
        batch_sequence_length_list.append(len(_input))

        # Inputs are padded.
        input_pad = [data_utils.PAD_ID] * (input_size - len(_input))
        inputs.append(list(_input + input_pad))

        # labels don't need padding.
        s_attrs.append(_s_attr)
        s_locs.append(_s_loc)
        s_names.append(_s_name)
        s_opes.append(_s_ope)
        s_ways.append(_s_way)
        intents.append(_intent)

        # Now we create batch-major vectors from the data selected above.
        batch_inputs = np.array(inputs, dtype=np.int32)

        def one_hot(vector, num_classes):
            assert isinstance(vector, np.ndarray)
            assert len(vector) > 0

            if num_classes is None:
                num_classes = np.max(vector) + 1
            else:
                assert num_classes > 0
                assert num_classes >= np.max(vector)

            result = np.zeros(shape=(len(vector), num_classes))
            result[np.arange(len(vector)), vector] = 1
            return result.astype(int)

        batch_s_attrs = one_hot(np.array([s_attrs[batch_idx][0] for batch_idx in range(1)],
                                         dtype=np.int32), s_attr_size)
        batch_s_locs = one_hot(np.array([s_locs[batch_idx][0] for batch_idx in range(1)],
                                        dtype=np.int32), s_loc_size)
        batch_s_names = one_hot(np.array([s_names[batch_idx][0] for batch_idx in range(1)],
                                         dtype=np.int32), s_name_size)
        batch_s_opes = one_hot(np.array([s_opes[batch_idx][0] for batch_idx in range(1)],
                                        dtype=np.int32), s_ope_size)
        batch_s_ways = one_hot(np.array([s_ways[batch_idx][0] for batch_idx in range(1)],
                                        dtype=np.int32), s_way_size)
        batch_intents = one_hot(np.array([intents[batch_idx][0] for batch_idx in range(1)],
                                         dtype=np.int32), intent_size)

        batch_sequence_length = np.array(batch_sequence_length_list, dtype=np.int32)
        # print('batch_inputs', batch_inputs)
        # print('batch_s_attrs', batch_s_attrs)

        return batch_inputs, batch_s_attrs, batch_s_locs, batch_s_names, batch_s_opes, batch_s_ways, batch_intents, batch_sequence_length
