import tensorflow as tf
import bucketmodel

VOCAB = {
    '<UNK>': 0,
    '<PAD>': 1,
    '<GO>': 2,
    '<EOS>': 3,
    'hello': 4,
    'world': 5,
    'cover': 6,
    'me': 7,
    'hi': 8,
    'roger': 9,
}

REVERSE_VOCAB = dict((v, k) for (k, v) in VOCAB.iteritems())


def test_seq2seq_pad():
    # Test with the following translation pairs
    # ["hello", "world"] -> ["<GO>", "hi", "<EOS>"]
    # ["cover", "me"] -> ["<GO>", "roger", "<EOS>"]
    encoder_input, decoder_input = bucketmodel.seq2seq_pad(
        [['hello', 'world'], ['cover', 'me']], 4,
        [['<GO>', 'hi', '<EOS>'], ['<GO>', 'roger', '<EOS>']], 5, VOCAB, VOCAB)
    assert list(encoder_input) == [[1, 1, 4, 5], [1, 1, 6, 7]]
    assert list(decoder_input) == [[2, 8, 3, 1, 1], [2, 9, 3, 1, 1]]


def test_bucketseq2seq_read_data():
    buckets = [(4, 4), (6, 6), (10, 10)]
    model = bucketmodel.BucketSeq2Seq(buckets, len(VOCAB), len(VOCAB), 128,
                                      128)
    #
    data = [
        (('hello', 'world'), ('hi', )),  # should be put into bucket (4, 4)
        (('cover', 'me'), ('roger', )),  # should be put into bucket (4, 4)
        (
            ('hello', 'hello'), ('hi', 'world', 'me', 'hi')
        )  # should be put into bucket (6, 6) after <GO> added to decoder_input
    ]
    model.read_data(data, VOCAB, VOCAB)
    assert model.buckets[0] == (4, 4)
    # batch_size == 2
    assert len(model.bucket_id_to_data[0][0][0]) == 2
    assert model.buckets[1] == (6, 6)
    # batch_size == 1
    assert len(model.bucket_id_to_data[1][0][0]) == 1

    bucket0_data = model.bucket_id_to_data[0]
    # encoder_input
    assert bucket0_data[0] == [(VOCAB['<PAD>'], VOCAB['<PAD>']),
                               (VOCAB['<PAD>'], VOCAB['<PAD>']),
                               (VOCAB['hello'], VOCAB['cover']),
                               (VOCAB['world'], VOCAB['me'])]
    # decoder_input
    # there is one more input for target left shift
    assert bucket0_data[1] == [(VOCAB['<GO>'], VOCAB['<GO>']),
                               (VOCAB['hi'], VOCAB['roger']),
                               (VOCAB['<EOS>'], VOCAB['<EOS>']),
                               (VOCAB['<PAD>'], VOCAB['<PAD>']),
                               (VOCAB['<PAD>'], VOCAB['<PAD>'])]


def test_train():
    tf.reset_default_graph()
    tf.set_random_seed(0)
    buckets = [(4, 4), (6, 6), (10, 10)]
    model = bucketmodel.BucketSeq2Seq(buckets, len(VOCAB), len(VOCAB), 128,
                                      128)
    #
    data = [(('hello', 'world'), ('hi', )), (('cover', 'me'), ('roger', )),
            (('hello', 'hello'), ('hi', 'world', 'me'))]
    model.create_model(tf.train.GradientDescentOptimizer(0.01))
    model.read_data(data, VOCAB, VOCAB)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    loss_with_bucket0 = []

    def callback(collects):
        if collects['bucketid'] == 0:
            loss_with_bucket0.append(collects['loss'])

    model.train(sess, 50, callback)
    # Probabilistic testing. Generally the cost will reduce
    assert loss_with_bucket0[0] > loss_with_bucket0[24]
    assert loss_with_bucket0[24] > loss_with_bucket0[-1]
    sess.close()


def _collect_decode_outputs(outputs):
    decode_res = dict()
    for o in outputs:
        if o is not None:
            for (encoder_words, decoder_words
                 ) in bucketmodel.encoder_decoder_ids_to_word_pairs(
                     o[0], o[1], REVERSE_VOCAB, REVERSE_VOCAB):
                decode_res[encoder_words] = decoder_words
    return decode_res


def test_predict():
    tf.reset_default_graph()
    tf.set_random_seed(0)
    buckets = [(4, 4), (6, 6), (10, 10)]
    model = bucketmodel.BucketSeq2Seq(buckets, len(VOCAB), len(VOCAB), 128,
                                      128)
    #
    data = [(('hello', 'world'), ('hi', )), (('cover', 'me'), ('roger', )),
            (('hello', 'hello'), ('hi', 'world', 'me'))]
    model.create_model(tf.train.AdagradOptimizer(0.1))
    model.read_data(data, VOCAB, VOCAB)
    sess = tf.Session()
    # We shouldn't be doing well before training
    sess.run(tf.global_variables_initializer())
    outputs = model.decode(sess)
    decode_res = _collect_decode_outputs(outputs)
    assert decode_res[('hello', 'hello')] != ('hi', 'world', 'me')

    # Train
    model.train(sess, 50)

    # We should be doing well after training
    outputs = model.decode(sess)
    decode_res = _collect_decode_outputs(outputs)
    assert decode_res[('hello', 'world')] == ('hi', )
    assert decode_res[('cover', 'me')] == ('roger', )
    assert decode_res[('hello', 'hello')] == ('hi', 'world', 'me')
    sess.close()
