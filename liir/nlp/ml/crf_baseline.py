from nltk.tag import CRFTagger
import liir.nlp.preprocessing as P
import sys


def run_crf(trainfile, testfile, model_file=None):

    maxlen = 100
    sents_train, tags_train, unique_words_train, unique_tags_train = \
        P.retrieve_sentences_tags(trainfile, maxlen=maxlen)
    sents_test, tags_test, unique_word_test, unique_tags_test = \
        P.retrieve_sentences_tags(testfile, maxlen=maxlen, allowedtags=unique_tags_train)

    train_data = []
    for n, st in enumerate(sents_train):
        s = []
        for m, _ in enumerate(st):
            s.append((unicode(sents_train[n][m], "utf-8")
                      , unicode(tags_train[n][m], "utf-8")))
        train_data.append(s)

    crf = CRFTagger()
    if model_file is None:
        crf.train(train_data, model_file='data/crf.mdl')
    else:
        crf.set_model_file(model_file)

    test_data = []
    for n, st in enumerate(sents_test):
        s = []
        for m, _ in enumerate(st):
            s.append((unicode(sents_test[n][m], "utf-8")
                      , unicode(tags_test[n][m], "utf-8")))
        test_data.append(s)

    print(crf.evaluate(test_data))

if __name__ == "__main__":

    TRAINFILE = 'data/conll_train_full_processed.txt'

    try:
        TESTFILE = sys.argv[1]
    except IndexError as e:
        print("Must specify file to test")
        raise e

    try:
        MODFILE = sys.argv[2]
    except IndexError:
        MODFILE = None
        print('Training fresh model')
        pass

    run_crf(TRAINFILE, TESTFILE, MODFILE)
