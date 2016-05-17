import os
import socket
import sys

if socket.gethostname() == 'bilbo':
    sys.path.remove('/usr/lib/python2.7/dist-packages')
    sys.path.append('/usr/lib/python2.7/dist-packages')
elif socket.gethostname() == 'tedz-hp':
    sys.path.append('/home/tedz/Desktop/research/SCL')

import gensim
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM, GRU
from keras.layers import TimeDistributed
from keras.utils.generic_utils import Progbar
import liir.nlp.preprocessing as P


def custom_accuracy(y_true, y_pred):
    """
    Calculate accuracy by discarding predictions for outputs
    whose true value is a padding (0's)
    :param y_true:
    :param y_pred:
    :return:
    """
    assert len(y_true) == len(y_pred)
    n_correct = 0
    n_incorrect = 0
    length = len(y_true[0])
    rev = range(0, length)
    rev.reverse()
    for n, y in enumerate(y_true):
        hypo = y_pred[n]
        for i in rev:
            if y[i] == 0:
                break
            elif y[i] == hypo[i]:
                n_correct += 1
            else:
                n_incorrect += 1
    return n_correct, n_incorrect


def batch(x_data, y_data, vocab_dim, embedding_weights=None, n=1024, shuffle=False):
    """
    batchify training examples, so that not all of them need to be loaded into
    memory at once
    :param x_data: all x data in indices
    :param y_data: tags
    :param vocab_dim: vocabulary dimension
    :param embedding_weights: mapping from word indices to word embeddings
    :param n: batch size
    :param shuffle: whether or not to shuffle them
    :return: word embeddings corresponding to x indices, and corresponding tags
    """
    l = len(x_data)
    # shuffle the data
    if shuffle:
        randIndices = np.random.permutation(l)
        x_data = np.array([x_data[i] for i in randIndices])
        y_data = np.array([y_data[i] for i in randIndices])

    for ndx in range(0, l, n):
        x_data_subset = x_data[ndx:min(ndx + n, l)]
        y_data_subset = y_data[ndx:min(ndx + n, l)]
        x_out = np.zeros([len(x_data_subset), x_data.shape[1], vocab_dim])
        for i, example in enumerate(x_data_subset):
            for j, word in enumerate(example):
                x_out[i][j] = embedding_weights[word]
        yield x_out, y_data_subset


def run_training(trainfile, testfile, embeddings_file, epochs,
                 maxlen=100,
                 batch_size=1024):
    print('Loading data...')
    sents_train, truths_train, unique_words_train, unique_tags_train = \
        P.retrieve_sentences_tags(trainfile, maxlen=maxlen)
    sents_test, truths_test, unique_word_test, unique_tags_test = \
        P.retrieve_sentences_tags(testfile, maxlen=maxlen, allowedtags=unique_tags_train)

    alltags = unique_tags_train.union(unique_tags_test)
    uniqueWords = unique_words_train.union(unique_word_test)

    gsm_mod = gensim.models.Word2Vec.load_word2vec_format(embeddings_file)
    vocab_dim = len(gsm_mod['word'])

    tagDict = {}
    for n, t in enumerate(alltags):
        tagDict[t] = n + 1

    index_dict = {}
    for n, word in enumerate(uniqueWords):
        index_dict[word] = n + 1

    nb_classes = len(tagDict)

    X_train, Y_train = P.create_input_data(sents_train, truths_train, index_dict,
                                           tagDict, maxlen=maxlen)
    X_test, Y_test = P.create_input_data(sents_test, truths_test, index_dict,
                                         tagDict, maxlen=maxlen)

    # makes output classes binary vectors instead of class numbers
    Y_train_cat = np.array([to_categorical(y, nb_classes=nb_classes + 1) for y in Y_train])
    Y_test_cat = np.array([to_categorical(y, nb_classes=nb_classes + 1) for y in Y_test])

    print(Y_train_cat.shape)
    print(X_train.shape)

    n_symbols = len(uniqueWords) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = gsm_mod[word]

    # assemble the model
    model = Sequential()  # or Graph or whatever
    model.add(LSTM(128, return_sequences=True, input_shape=(maxlen, vocab_dim)))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(nb_classes + 1)))
    model.add(Activation('softmax'))

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    keep_iterating = True
    count = 0
    cwd = os.getcwd()
    while keep_iterating:
        # making sure to not save the weights as the same name as
        # another is using
        count += 1
        tmpweights = "{}/tmp/weights{}.hdf5".format(cwd, count)
        if not os.path.isfile(tmpweights):
            keep_iterating = False

    print('============Training Params============\n'
          'Training file: {}\nTesting file: {}\nEpochs: {}\n'
          'Max length of sentence: {}\nWord embedding dimensions: {}\n'
          'Batch size: {}\n'
          '======================================='
          .format(trainfile, testfile, epochs, maxlen, vocab_dim, batch_size))

    print('Train...')
    best_yet = 0
    for e in range(epochs):
        print("Training epoch {}".format(e + 1))
        pbar = Progbar(1 + len(X_train)/batch_size)
        count = 0
        for xt, yt in batch(X_train, Y_train_cat, vocab_dim, embedding_weights, n=batch_size, shuffle=True):
            count += 1
            model.fit(xt, yt, batch_size=batch_size, nb_epoch=1, verbose=False)
            pbar.update(count)

        # free up some space
        xt = None
        yt = None

        print("Training finished, evaluating on {} validation samples".format(batch_size))
        # take a random subset of validation data
        for X_test_subset, Y_test_subset in batch(X_test, Y_test, n=batch_size, shuffle=True):
            hypo = model.predict_classes(X_test_subset, batch_size=1)
            break

        correct, incorrect = custom_accuracy(y_true=Y_test_subset, y_pred=hypo)
        acc = correct / float(correct + incorrect)
        print("Correct: {}\nIncorrect: {}\n Accuracy: {}"
              .format(correct, incorrect, acc))
        if acc > best_yet:
            print('Improved from {} to {}, saving weights to {}\nEpoch {} finished.'
                  .format(best_yet, acc, tmpweights, e + 1))
            best_yet = acc
            model.save_weights(tmpweights, overwrite=True)

    model.load_weights(tmpweights)
    # evaluate on model's best weights

    first = True
    for xt, yt in batch(X_test, Y_test_cat, vocab_dim, embedding_weights, n=batch_size):
        hypo = model.predict_classes(xt, batch_size=1)
        if first:
            Y_hypo = hypo
            first = False
        else:
            Y_hypo = np.concatenate((Y_hypo, hypo))

    correct, incorrect = custom_accuracy(y_true=Y_test, y_pred=Y_hypo)
    print("Finished! Final Score\nCorrect: {}\nIncorrect: {}\n Accuracy: {}"
          .format(correct, incorrect, float(correct) / (correct + incorrect)))

    log = '{}/tmp/log_{}.txt'.format(cwd, count)
    f = open(log, 'w')
    f.write('Embeddings file: {}\n'.format(embeddings_file))
    f.close()
    print('Log saved as {}'.format(log))


if __name__ == "__main__":

    TRAINFILE = './data/conll_train_full_processed.txt'
    EPOCHS = 25

    try:
        TESTFILE = sys.argv[1]
    except IndexError as e:
        print("Must specify file to test")
        raise e

    try:
        EMBEDDINGSFILE = sys.argv[2]
    except IndexError as e:
        print("Must specify embeddings file to load")
        raise e

    try:
        EPOCHS = int(sys.argv[3])
    except IndexError:
        pass

    run_training(TRAINFILE, TESTFILE, EMBEDDINGSFILE, EPOCHS)
