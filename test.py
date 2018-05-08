# TensorFlow code as per tutorial
import tensorflow as tf


def create_file_reader_ops(filename_queue):
    reader = tf.TextLineReader(skip_header_lines=1)
    _, csv_row = reader.read(filename_queue)
    record_defaults = [[''], [''], [0], [0], [0], [0]]
    country, code, gold, silver, bronze, total = tf.decode_csv(csv_row, record_defaults=record_defaults)
    features = tf.stack([gold, silver, bronze])
    label = tf.stack([country])
    return features, label


filenames = ["olympics2016.csv"]
batch_size = 10

filename_queue = tf.train.string_input_producer(filenames, num_epochs=1, shuffle=False)
example, label = create_file_reader_ops(filename_queue)
model = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(model)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    while True:
        try:
            example, label = sess.run([example, label])
            print(example, label)
        except tf.errors.OutOfRangeError:
            break

    print("Out of Range")