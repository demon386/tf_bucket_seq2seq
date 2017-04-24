Bucket-based Tensorflow seq2seq model


This is a demo of Bucket-based Tensorflow seq2seq model, with supporting for both training and decoding.

Please see `test_predict` in `test_bucketmodel.py` for usage.

Internally, it uses `tf.contrib.legacy_seq2seq.model_with_buckets` to build multiple models for each bucket. It can handle data pre-process and data post-process for using bucket-based models.
