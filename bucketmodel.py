# Author: M. Tong <demon386@gmail.com>
from __future__ import print_function
import random
import tensorflow as tf
import itertools
import numpy as np
import sys
from collections import defaultdict
from tensorflow.contrib.rnn import BasicRNNCell
from tensorflow.contrib.legacy_seq2seq import embedding_rnn_seq2seq


def seq2seq_pad(encoder_inputs,
                encoder_length,
                decoder_inputs,
                decoder_length,
                source_vocab,
                target_vocab,
                pad_symbol='<PAD>',
                unk_symbol='<UNK>'):
    """
    - encoder_input: A nested list of symbol str for encoding, length: batch_size
    - encoder_length: max length of encoder input
    - decoder_input: A nested list of symbol str for decoding, length: batch_size
    - decoder_length: max length of decoder input
    - vocab: vocabulary index, symbol (str) -> index (int)

    Example:
    ["hello", "world"] -> ["<GO>", "hi", "<EOS>"]
    ["cover", "me"] -> ["<GO>", "roger", "<EOS>"]

    seq2seq_pad([['hello', 'world'], ['cover', 'me']], 4, [['<GO>', hi', '<EOS>'], ['<GO>', 'roger', '<EOS>']], 4, vocab)

    Assume that index of "<PAD>" is 0

    Output:
    [[0, 0, <index of 'hello'>, <index of 'world'>], [0, 0, <index of 'cover'>, <index of 'me'>]],
    [[<index of '<GO>'>, <index of 'hi'>, <index of 'EOS'>, 0, 0], [<index of '<GO>'>, <index of 'roger'>, <index of 'EOS'>, 0, 0]]
    """

    def to_index(inputs, length, vocab, pad_from_start=True):
        inputs_to_index = []
        unk_index = vocab[unk_symbol]
        for cur_input in inputs:
            cur_input_to_index = [vocab[pad_symbol]] * length
            l = len(cur_input)
            if l < length:
                if pad_from_start:
                    cur_input_to_index[(length - l):] = [
                        vocab.get(i, unk_index) for i in cur_input
                    ]
                else:
                    cur_input_to_index[:l] = [vocab.get(i, unk_index)
                                              for i in cur_input]
            else:
                cur_input_to_index = [vocab.get(i, unk_index)
                                      for i in cur_input[:length]]
            inputs_to_index.append(cur_input_to_index)
        return inputs_to_index

    return to_index(encoder_inputs, encoder_length, source_vocab,
                    True), to_index(decoder_inputs, decoder_length,
                                    target_vocab, False)


def encoder_decoder_ids_to_word_pairs(encoder_ids,
                                      decoder_ids,
                                      reverse_source_vocab,
                                      reverse_target_vocab,
                                      decoder_eos_symbol="<EOS>",
                                      pad_symbol="<PAD>"):
    """
    Tensorflow seq2seq input format back to (encoder_words, deocder_words) pairs
    """
    encoder_ids = zip(*encoder_ids)
    decoder_ids = zip(*decoder_ids)

    def ids_to_words(ids, reverse_vocab, remove_pad_form_start=True):
        words = (reverse_vocab[i] for i in ids)
        words = itertools.takewhile(lambda x: x != decoder_eos_symbol, words)
        if remove_pad_form_start:
            words = itertools.dropwhile(lambda x: x == pad_symbol, words)
        return tuple(words)

    encoder_words = map(
        lambda ids: ids_to_words(ids, reverse_source_vocab, True), encoder_ids)
    decoder_words = map(
        lambda ids: ids_to_words(ids, reverse_target_vocab, False),
        decoder_ids)

    for (encoder, decoder) in zip(encoder_words, decoder_words):
        yield (tuple(encoder), tuple(decoder))


class BucketSeq2Seq:
    def __init__(self, buckets, num_encoder_symbols, num_decoder_symbols,
                 embedding_dim, rnn_state_dim):
        """
        - buckets: A list of (encoder_length, decoder_length) bucket to put
          sample in, with ascending order
        - num_encoder_symbols: number of encoder symbols (for encoding word embedding)
        - num_decoder_symbols: number of decoder symbols (for decoding word embedding)
        - embedding_dim: embedding dimension for input & output
        - rnn_state_dim: state dimension for BasicRNNCell

        Example: BucketSeq2Seq([(5, 10), (10, 15), (20, 25), (40, 50)])
        """
        self.buckets = buckets
        self.num_encoder_symbols = num_encoder_symbols
        self.num_decoder_symbols = num_decoder_symbols
        self.embedding_dim = embedding_dim
        self.rnn_state_dim = rnn_state_dim

    def _find_bucket_id(self, encoder_size, decoder_size):
        """
        Find the smallest bucket id compatible for the given encoder and decoder size

        If not found, return None
        """
        for bucket_id, (source_size, target_size) in enumerate(self.buckets):
            if encoder_size <= source_size and decoder_size <= target_size:
                return bucket_id
        return None

    def _tf_seq2seq_model(self,
                          encoder_inputs,
                          decoder_inputs,
                          do_decode=False):
        model = embedding_rnn_seq2seq(encoder_inputs,
                                      decoder_inputs,
                                      self.cell,
                                      self.num_encoder_symbols,
                                      self.num_decoder_symbols,
                                      self.embedding_dim,
                                      output_projection=None,
                                      feed_previous=do_decode)

        if hasattr(self.cell, '_scope'):
            print(self.cell._scope.name)
        return model

    def create_model(self, optimizer):
        max_encoder_size = self.buckets[-1][0]
        max_decoder_size = self.buckets[-1][1]
        self.encoder_placeholders = [tf.placeholder(tf.int32,
                                                    shape=[None],
                                                    name="encoder_%d" % i)
                                     for i in range(max_encoder_size)]
        self.decoder_placeholders = [tf.placeholder(tf.int32,
                                                    shape=[None],
                                                    name="decoder_%d" % i)
                                     for i in range(max_decoder_size + 1)]
        # number of weights is one less than decoder_placeholders
        self.target_weights_placeholders = [tf.placeholder(
            tf.float32,
            shape=[None],
            name="weight_%d" % i) for i in range(max_decoder_size)]

        self.cell = BasicRNNCell(self.rnn_state_dim)
        self.encoder_cell = BasicRNNCell(self.rnn_state_dim)

        # Example:
        # decoder_inputs:  <GO>, hello, <EOS>
        # targets:        hello, <EOS>
        # The last symbol in decoder_inputs is only used for targets
        targets = self.decoder_placeholders[1:]
        self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
            self.encoder_placeholders, self.decoder_placeholders[:-1], targets,
            self.target_weights_placeholders, self.buckets,
            lambda x, y: self._tf_seq2seq_model(x, y, False))
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self.predict_outputs, _ = tf.contrib.legacy_seq2seq.model_with_buckets(
                self.encoder_placeholders, self.decoder_placeholders[:-1],
                targets, self.target_weights_placeholders, self.buckets,
                lambda x, y: self._tf_seq2seq_model(x, y, True))
        self.optimizers = [optimizer.minimize(i) for i in self.losses]

    def read_data(self,
                  data,
                  source_vocab,
                  target_vocab,
                  decoder_start_symbol="<GO>",
                  decoder_eos_symbol="<EOS>",
                  pad_symbol="<PAD>",
                  unk_symbol="<UNK>"):
        """
        data: Iterable, the format of the item is ("source_word1", "source_word2") -> ("target_word1", "target_word2", "target_word3")

        The function will process the input:

        For decoder input:
        - prepend "<GO>" to decoder symbols
        - apend "<EOS>" and "<PAD>" to decoder symbols

        For encoder input:
        - prepend "<PAD>" to encoder symbols
        """

        self.bucket_id_to_data = defaultdict(list)
        self.target_pad_idx = target_vocab[pad_symbol]
        for (raw_encoder_input, raw_decoder_input) in data:
            encoder_length = len(raw_encoder_input)
            # raw length + <GO> and <EOS>
            decoder_length = len(raw_decoder_input) + 1
            bucket_id = self._find_bucket_id(encoder_length, decoder_length)
            if bucket_id is None:
                print("Failed to find compatible bucket for input: (%s, %s)." %
                      (' '.join(raw_encoder_input),
                       ' '.join(raw_decoder_input)),
                      file=sys.stderr)
                continue
            # we will prepend "<GO>" and "<EOS> later"
            decoder_input_with_start_and_end = [
                decoder_start_symbol
            ] + list(raw_decoder_input) + [decoder_eos_symbol]
            self.bucket_id_to_data[bucket_id].append((
                raw_encoder_input, decoder_input_with_start_and_end))

        # pad the bucket
        for (bucket_id, val) in self.bucket_id_to_data.iteritems():
            encoder_length, decoder_length = self.buckets[bucket_id]
            encoder_inputs, decoder_inputs = zip(*val)
            inputs_with_pad = seq2seq_pad(encoder_inputs, encoder_length,
                                          decoder_inputs, decoder_length + 1,
                                          source_vocab, target_vocab,
                                          pad_symbol, unk_symbol)
            self.bucket_id_to_data[bucket_id] = (zip(*inputs_with_pad[0]),
                                                 zip(*inputs_with_pad[1]))

    def train(self, session, num_iteration, callback=None):
        # sort buckets by random
        buckets_with_data = list(self.bucket_id_to_data.iterkeys())
        random.shuffle(buckets_with_data)

        bucketid_to_feed_dict = dict()
        for bucket_id in buckets_with_data:
            # pick out optimizer from the corresponding seq2seq model
            feed_dict = self._build_feed_dict(bucket_id)
            bucketid_to_feed_dict[bucket_id] = feed_dict

        for iteration in range(num_iteration):
            for bucket_id in buckets_with_data:
                feed_dict = bucketid_to_feed_dict[bucket_id]
                optimizer = self.optimizers[bucket_id]
                loss = self.losses[bucket_id]
                _, loss_val = session.run([optimizer, loss], feed_dict)
                if callback:
                    callback({'iteration': iteration,
                              'bucketid': bucket_id,
                              'loss': loss_val})

    def decode(self, session, by_bucket=True):
        """
        - by_bucket: decode by bucket, will not guarantee the order, but
          generally with better performance
        """
        # @todo: support by_bucket=False
        buckets_with_data = list(self.bucket_id_to_data.iterkeys())
        # (input, output, prob_matrix)
        output_tuples = [None] * len(self.buckets)
        for bucket_id in buckets_with_data:
            # pick out optimizer from the corresponding seq2seq model
            feed_dict = self._build_feed_dict(bucket_id)
            encoder_inputs = self.bucket_id_to_data[bucket_id][0]
            predict_output = self.predict_outputs[bucket_id]
            predict_labels = [tf.argmax(i, axis=1) for i in predict_output]
            predict_labels_val, prob_matrices = session.run(
                [predict_labels, predict_output], feed_dict)
            output_tuples[bucket_id] = (encoder_inputs, predict_labels_val,
                                        prob_matrices)
        return output_tuples

    def _build_feed_dict(self, bucket_id):
        encoder_inputs, decoder_inputs = self.bucket_id_to_data[bucket_id]
        feed_dict = {}
        # encoder
        for (placeholder, input_) in zip(self.encoder_placeholders,
                                         encoder_inputs):
            feed_dict[placeholder] = input_

        # decoder
        for (placeholder, input_) in zip(self.decoder_placeholders,
                                         decoder_inputs):
            feed_dict[placeholder] = input_

        # target weights
        for (placeholder, input_) in zip(self.target_weights_placeholders,
                                         decoder_inputs[1:]):
            # remember to shift by one
            feed_dict[placeholder] = np.asarray(input_) != self.target_pad_idx
        return feed_dict
