"""
This file processes different datasets into the same format so that
in training and evaluations we can easily switch between them
"""


def genia_split(line, secondTag=False):
    """
    In genia corpus, some words have two tags associated like so:
    Some words have two tag associated, currently we take
    the first tag. To take the second tag, uncomment  the
    line indicated below
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
    processes
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
