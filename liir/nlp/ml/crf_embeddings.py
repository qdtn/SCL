import sys

import gensim
import pycrfsuite
from nltk.tag import CRFTagger

import liir.nlp.preprocessing as P


class CRFModel(object):
    def __init__(self, c1=1.0, c2=1e-3,
                 max_iterations=50,
                 feature_possible_transitions=True
                 ):
        self.classifier = None
        self.model_path = None
        self.c1 = c1
        self.c2 = c2
        self.max_iterations = max_iterations
        self.feature_possible_transitions = feature_possible_transitions

    def train(self, X, Y, model_path=None):
        trainer = pycrfsuite.Trainer(verbose=False)

        for xseq, yseq in zip(X, Y):
            trainer.append(xseq, yseq)
        if model_path is None:
            model_path = "data/crf_embeddings.mdl"
        trainer.train(model_path)
        self.model_path = model_path
        return

    def set_model_path(self, model_path):
        self.model_path = model_path

    def predict(self, X):
        if self.classifier is None:
            self.classifier = pycrfsuite.Tagger()
            self.classifier.open(self.model_path)

        Ypredict = []
        for xseq in X:
            self.classifier.set(xseq)
            yseq = self.classifier.tag()
            #  for i in range(len(yseq)):
            #      print(tagger.marginal(yseq[i],i))
            Ypredict.append(yseq)
        return Ypredict

    def evaluate(self, X, TRUTH):
        correct = 0
        incorrect = 0
        predictions = self.predict(X)
        for xpred, truth in zip(predictions, TRUTH):
            for x, t in zip(xpred, truth):
                if x == t:
                    correct += 1
                else:
                    incorrect += 1
        acc = correct / float(correct + incorrect)
        return acc


def convert(X):
    """
    converts training data X into a format that pycrf
    accepts, since typically it accepts strings only
    :param X: word embeddings, 3D
    :return:
    """
    X_crf = []
    for seq in X:
        sq_dt = []
        for ins in seq:
            indices = {}
            for i in range(len(ins)):
                indices[str(i)] = ins[i]
            sq_dt.append(indices)
        iq = pycrfsuite.ItemSequence(sq_dt)
        X_crf.append(iq)
    return X_crf


def run_crf(trainfile, testfile, embeddingsfile, model_file=None):
    maxlen = 100
    sents_train, tags_train, unique_words_train, unique_tags_train = \
        P.retrieve_sentences_tags(trainfile, maxlen=maxlen)
    sents_test, tags_test, unique_word_test, unique_tags_test = \
        P.retrieve_sentences_tags(testfile, maxlen=maxlen, allowedtags=unique_tags_train)
    gsm_mod = gensim.models.Word2Vec.load_word2vec_format(embeddingsfile)

    for n, sent in enumerate(sents_test):
        myvec = [gsm_mod[word] for word in sent]
        sents_test[n] = myvec
    sents_test = convert(sents_test)

    crf = CRFTagger()
    if model_file is None:
        for n, sent in enumerate(sents_train):
            # convert input vectors from words to word embeddings
            myvec = [gsm_mod[word] for word in sent]
            sents_train[n] = myvec
        sents_train = convert(sents_train)
        crf.train(sents_train, tags_train, model_file='data/crf_embeddings.mdl')
    else:
        crf.set_model_file(model_file)

    print(crf.evaluate(sents_test, tags_test))


if __name__ == "__main__":

    TRAINFILE = 'data/conll_train_full_processed.txt'

    try:
        TESTFILE = sys.argv[1]
    except IndexError as e:
        print("Must specify file to test")
        raise e

    try:
        EMBEDDINGSFILE = sys.argv[2]
    except IndexError as e:
        print("Must specify embeddings to use")
        raise e

    try:
        MODFILE = sys.argv[3]
    except IndexError:
        MODFILE = None
        print('Training fresh model')
        pass

    run_crf(TRAINFILE, TESTFILE, EMBEDDINGSFILE, MODFILE)
