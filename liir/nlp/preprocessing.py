"""
This file processes different datasets into the same format so that
in training and evaluations we can easily switch between them
"""


def genia_split(line, secondTag=False):
    """
    In genia corpus, some words have two tags associated like so:
    'acetyl/JJ|NN'. The default option is to take the first tag,
    To take the second tag, set argument to True
    :param line:
    :return:
    """
    l = len(line) - 1
    idx1 = l + 1
    tag = None
    for n in range(l, -1, -1):
        if line[n] == '|':
            if secondTag:
                tag = line[n + 1:idx1]
            # ^uncomment line to take second tag
            idx1 = n
        if line[n] == '/':
            tag = line[n + 1:idx1] if tag is None else tag
            word = line[0:n]
            return word, tag
    return None, None


def process_genia_dataset(infile, outfile='data/genia_processed.txt'):
    """
    Processes genia dataset. Outputs a file with 3 columns:
    nth word in sentece, word, POS tag
    :param infile:
    :param outfile:
    :return:
    """
    sentences = []
    alltags = []
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for line in f:
            line = line.rstrip('\n')
            if line == '====================' or line == '\n':
                sentences.append(new_sentence)
                alltags.append(new_tags)
                new_sentence = []
                new_tags = []
                continue
            word, tag = genia_split(line)
            if word is None:
                continue
            new_sentence.append(word)
            new_tags.append(tag)
    f = open(outfile, 'w')
    for m, sent in enumerate(sentences):
        tags = alltags[m]
        for n, word in enumerate(sent):
            f.write('{}\t{}\t{}\n'.format(n + 1, word, tags[n]))
        f.write('\n')
    f.close()
    return


def process_conll_dataset(infile, outfile='data/conll_processed.txt'):
    """
    Processes conll dataset. Outputs a file with 3 columns:
    nth word in sentece, word, POS tag
    :param infile:
    :param outfile:
    :return:
    """
    sentences = []
    alltags = []
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for line in f:
            splits = line.split('\t')
            if not splits[0].isdigit():
                sentences.append(new_sentence)
                alltags.append(new_tags)
                new_sentence = []
                new_tags = []
                continue
            tag = splits[4]
            word = splits[1]
            new_tags.append(tag)
            new_sentence.append(word)
    f = open(outfile, 'w')
    for m, sent in enumerate(sentences):
        tags = alltags[m]
        for n, word in enumerate(sent):
            f.write('{}\t{}\t{}\n'.format(n + 1, word, tags[n]))
        f.write('\n')
    f.close()
    return


def retrieve_sentences_tags(infile):
    """
    Loads files processed according to the formats in other
    functions of this file into variables
    :param infile:
    :return: sentences, tags corresponding to each word in each sentence,
    set of unique words used contained in dataset, and set of unique tags
    contained in dataset
    """
    sents = []
    truths = []
    words = set([])
    tags = set([])
    with open(infile, 'r') as f:
        new_sentence = []
        new_tags = []
        for n, line in enumerate(f):
            splits = line.split('\t')
            if not splits[0].isdigit():
                sents.append(new_sentence)
                truths.append(new_tags)
                new_sentence = []
                new_tags = []
                continue
            splits = line.split('\t')
            tag = splits[2]
            word_lower = splits[1].lower()
            tags.add(tag)
            new_tags.append(tag)
            words.add(word_lower)
            new_sentence.append(word_lower)
    return sents, truths, words, tags
