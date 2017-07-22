import os
import string
import tarfile

import collections
import requests
from nltk.corpus import stopwords
import numpy as np
import tensorflow as tf

vocabulary_size = 10000
embedding_size = 200
batch_size = 100
num_sampled = int(batch_size/2)
window_size = 2

valid_words = ['cliche', 'love', 'hate', 'silly', 'sad']


def load_movie_data():
    save_folder_name = 'temp'
    pos_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.pos')
    neg_file = os.path.join(save_folder_name, 'rt-polaritydata', 'rt-polarity.neg')

    if not os.path.exists(os.path.join(save_folder_name, 'rt-polaritydata')):
        movie_data_url = 'http://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
        req = requests.get(movie_data_url, stream=True)
        with open('temp_movie_review_temp.tar.gz', 'wb') as f:
            for chunk in req.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    f.flush()

        tar = tarfile.open('temp_movie_review_temp.tar.gz', 'r:gz')
        tar.extractall(path='temp')
        tar.close()

    pos_data = []
    with open(pos_file, 'r', encoding='latin-1') as f:
        for line in f:
            pos_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    pos_data = [x.rstrip() for x in pos_data]

    neg_data = []
    with open(neg_file, 'r', encoding='latin-1') as f:
        for line in f:
            neg_data.append(line.encode('ascii', errors='ignore').decode())
    f.close()
    neg_data = [x.rstrip() for x in neg_data]

    texts = pos_data + neg_data
    target = [1] * len(pos_data) + [0] *len(neg_data)
    return texts, target

# length is 10662
texts, target = load_movie_data()


def normalize_text(text):
    text = [x.lower() for x in text]
    text = [''.join(c for c in x if c not in '0123456789') for x in text]
    text = [''.join(c for c in x if c not in string.punctuation) for x in text]
    text = [' '.join([word for word in x.split() if word not in stopwords.words('english')]) for x in text]
    text = [' '.join(x.split()) for x in text]
    return text

texts = normalize_text(texts)

# after this operation the length is 10406
target = [target[ix] for ix, x in enumerate(texts) if len(x.split()) > 2]
texts = [x for x in texts if len(x.split()) > 2]


def build_dictionary(sentences, vocabulary_size):
    split_sentences = [x.split() for x in sentences]
    words = [x for sublist in split_sentences for x in sublist]

    word_count = [['RARE', -1]]
    word_count.extend(collections.Counter(words).most_common(vocabulary_size-1))

    word_dict = {}
    for word, count in word_count:
        # we only want a Id of every word, useless to store the frequency of every common words
        word_dict[word] = len(word_dict)

    return word_dict


def text_to_number(sentences, word_dict):
    data = []
    for sentence in sentences:
        sen_data = []
        for word in sentence.split(' '):
            if word in word_dict:
                x = word_dict[word]
            else:
                x = 0
            sen_data.append(x)
        data.append(sen_data)
    return data

word_dictionary = build_dictionary(texts, vocabulary_size)
word_dictionary_rev = dict(zip(word_dictionary.values(), word_dictionary.keys()))
text_data = text_to_number(texts, word_dictionary)

valid_examples = [word_dictionary[x] for x in valid_words]


def generate_batch_data(sentences, batch_size, window_size, method='skip_gram'):
    batch_data = []
    label_data = []
    while len(batch_data) < batch_size:
        rand_sentence = np.random.choice(sentences)
        window_sequence = [rand_sentence[max((ix-window_size), 0):(ix+window_size+1)]
                           for ix, x in enumerate(rand_sentence)]
        label_indices = [ix if ix < window_size else window_size for ix, x in enumerate(window_sequence)]
        if method == 'skip_gram':
            batch_and_labels = [(x[y], x[:y] + x[(y+1):]) for x, y in zip(window_sequence, label_indices)]
            tuple_data = [(x, y_) for x, y in batch_and_labels for y_ in y]
        else:
            raise ValueError('Method {} not implemented yet.'.format(method))

        batch, label = [list(x) for x in zip(*tuple_data)]
        batch_data.extend(batch[:batch_size])
        label_data.extend(label[:batch_size])
    batch_data = np.array(batch_data[:batch_size])
    label_data = np.transpose(np.array([label_data[:batch_size]]))
    return batch_data, label_data

embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

x_inputs = tf.placeholder(tf.int32, shape=[batch_size])
y_target = tf.placeholder(tf.int32, shape=[batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

embed = tf.nn.embedding_lookup(embeddings, x_inputs)

loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                     labels=y_target, inputs=embed,
                                     num_sampled=num_sampled,
                                     num_classes=vocabulary_size))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

loss_vec = []
loss_x_vec = []
for i in range(10000):
    batch_inputs, batch_labels = generate_batch_data(text_data,
                                                     batch_size, window_size)
    feed_dict = {x_inputs: batch_inputs, y_target: batch_labels}
    sess.run(optimizer, feed_dict=feed_dict)

    if (i+1) % 1000 == 0:
        loss_val = sess.run(loss, feed_dict=feed_dict)
        loss_vec.append(loss_val)
        loss_x_vec.append(i+1)
        print('Loss at step {} : {}'.format(i+1, loss_val))

    if (i+1) % 1000 == 0:
        sim = sess.run(similarity, feed_dict=feed_dict)
        for j in range(len(valid_words)):
            valid_word = word_dictionary_rev[valid_examples[j]]
            topk = 5
            nearest = (-sim[j,:]).argsort()[1:topk+1]
            log_str = "Nearest to {}:".format(valid_word)
            for k in range(topk):
                close_word = word_dictionary_rev[nearest[k]]
                log_str = "%s %s," % (log_str, close_word)
            print(log_str)

# Loss at step 10000 : 4.206537246704102
# Nearest to cliche: irony, shop, platitudes, ensure, fiction,
# Nearest to love: shore, visual, RARE, heres, brash,
# Nearest to hate: strangely, boisterous, scams, dynamic, questions,
# Nearest to silly: dazzling, superman, thcentury, smash, gollums,
# Nearest to sad: contrivance, ultraviolent, bed, coherence, phoniness,
