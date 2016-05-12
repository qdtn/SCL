import socket

if socket.gethostname() == 'bilbo':
    import sys

    sys.path.remove('/usr/lib/python2.7/dist-packages')
    sys.path.append('/usr/lib/python2.7/dist-packages')
import gensim
import numpy as np
from keras.preprocessing import sequence
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
import liir.nlp.preprocessing as P


def create_input_data(data_x, data_y, x_dict, y_dict, maxlen):
    """
    Creates data to be fed into neural network
    :param data_x: sentences in list form, like [['he', 'is', 'jolly'],['she',...]]
    :param data_y: tags corresponding to data_y
    :param x_dict: dictionary that maps words to indices (integers)
    :param y_dict: dictionary that maps tags to indices (integers)
    :param maxlen: maximum length of a sentence so we know how much padding to use
    :return: x, y that can be fed to the embedding layer of an LSTM
    """
    X_train = []
    Y_train = []
    for n, sent in enumerate(data_x):
        input = [x_dict[word] for word in sent]
        input = sequence.pad_sequences([input], maxlen=maxlen)
        output = [y_dict[tag] for tag in data_y[n]]
        output = sequence.pad_sequences([output], maxlen=maxlen)
        X_train.append(input[0])
        Y_train.append(output[0])
    return np.array(X_train), np.array(Y_train)


def custom_accuracy(y_true, y_pred):
    """
    Calculate accuracy by discarding predictions for outputs
    whose true value is a padding (0's)
    :param y_true:
    :param y_pred:
    :return:
    """
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


def run_training(trainfile, testfile, epochs,
                 maxlen=100,
                 vocab_dim=10,
                 batch_size=32):

    print("============Training Params============")
    print('Training file: {}\nTesting file: {}\nEpochs: {}\n'
          'Max length of sentence: {}\nWord embedding dimensions: {}\n'
          'Batch size: {}'.format(trainfile, testfile, epochs, maxlen,
                                  vocab_dim, batch_size))
    print("=======================================")

    sents_train, truths_train, unique_words_train, unique_tags_train = \
        P.retrieve_sentences_tags(trainfile)
    sents_test, truths_test, unique_word_test, unique_tags_test = \
        P.retrieve_sentences_tags(testfile)

    alltags = unique_tags_train.union(unique_tags_test)
    uniqueWords = unique_words_train.union(unique_word_test)
    gsm_mod = gensim.models.Word2Vec(sentences=sents_train + sents_test,
                                     size=vocab_dim, window=5, min_count=1,
                                     workers=4)
    gsm_mod.init_sims(replace=True)
    # saves ram when model is finished loading

    tagDict = {}
    for n, t in enumerate(alltags):
        tagDict[t] = n + 1

    index_dict = {}
    for n, word in enumerate(uniqueWords):
        index_dict[word] = n + 1

    nb_classes = len(tagDict)

    X_train, Y_train = create_input_data(sents_train, truths_train, index_dict,
                                         tagDict, maxlen=maxlen)
    X_test, Y_test = create_input_data(sents_test, truths_test, index_dict,
                                       tagDict, maxlen=maxlen)

    Y_train = np.array([to_categorical(y, nb_classes=nb_classes + 1) for y in Y_train])
    Y_test_cat = np.array([to_categorical(y, nb_classes=nb_classes + 1) for y in Y_test])

    print(Y_train.shape)
    print(X_train.shape)

    n_symbols = len(uniqueWords) + 1  # adding 1 to account for 0th index (for masking)
    embedding_weights = np.zeros((n_symbols, vocab_dim))
    for word, index in index_dict.items():
        embedding_weights[index, :] = gsm_mod[word]

    # assemble the model
    model = Sequential()  # or Graph or whatever
    model.add(Embedding(output_dim=vocab_dim, input_dim=n_symbols, mask_zero=True,
                        weights=[embedding_weights]))  # note you have to put embedding weights in a list by convention
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(nb_classes + 1)))
    model.add(Activation('softmax'))

    print('Loading data...')

    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print('Train...')
    #TODO: rewrite the training function to use correct losses during training
    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=epochs,
              validation_data=(X_test, Y_test_cat))
    score, acc = model.evaluate(X_test, Y_test_cat,
                                batch_size=batch_size)

    Y_hypo = model.predict_classes(X_test, batch_size=1)
    correct, incorrect = custom_accuracy(y_true=Y_test, y_pred=Y_hypo)
    print("Correct: {}\nIncorrect: {}\n Accuracy: {}"
          .format(correct, incorrect, float(correct) / (correct + incorrect)))
    print('Test score:', score)
    print('Test accuracy:', acc)


if __name__ == "__main__":

    TRAINFILE = './data/conll_train_full_processed.txt'
    TESTFILE = './data/conll_evaluation_ood_processed.txt'
    EPOCHS = 10
    try:
        TESTFILE = sys.argv[1]
    except IndexError:
        pass

    try:
        EPOCHS = int(sys.argv[2])
    except IndexError:
        pass

    run_training(TRAINFILE, TESTFILE, EPOCHS)
