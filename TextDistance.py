# Levenshtein distance: minimal steps to change one word to another

import tensorflow as tf

sess = tf.Session()


def create_sparse_vec(word_list):
    num_words = len(word_list)
    indices = [[xi, 0, yi] for xi, x in enumerate(word_list) for yi, y in enumerate(x)]
    chars = list(''.join(word_list))
    return tf.SparseTensorValue(indices, chars, [num_words, 1, 1])

hypothesis_words = ['bear','bar','tensor','flow']
truth_word = ['beers']

hyp_string_sparse = create_sparse_vec(hypothesis_words)
truth_string_sparse = create_sparse_vec(truth_word*len(hypothesis_words))

hyp_input = tf.sparse_placeholder(tf.string)
truth_input = tf.sparse_placeholder(tf.string)

edit_distances = tf.edit_distance(hyp_input, truth_input, normalize=True)

feed_dict = {
    hyp_input: hyp_string_sparse, truth_input: truth_string_sparse
}

print(sess.run(edit_distances, feed_dict=feed_dict))

