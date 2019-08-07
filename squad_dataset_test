import os
from similarity import rank_paragraphs, span_list, spacy_model
from pickle_docs import *
from tqdm import tqdm
from timeit import default_timer as timer


def preprocess_pickle(nlp, context_path, qas_path):
    """
    This process took 116.10 seconds
    :param nlp: the spacy model to process the text
    """
    print("pickling docs and qas")
    contexts, qas = json_context_qas_extract('/Users/mjeong/OneDrive - Infor/Downloads/train-v2.0.json', nlp)
    pickle_all(contexts,context_path)
    pickle_all(qas,qas_path)


def preprocess_unpickle(context_path, qas_path):
    """
    This process took 108.13 seconds, 88.38 seconds
    :return: list of contexts, list of tuples (target_paragraph_index, question spacy doc, answer text)
    """
    print("loading the pickled files")
    docs = unpickle_all(context_path)
    qas = unpickle_all(qas_path)
    return docs, qas


def answer_text2idx(contexts, qas):
    qas_idx = []
    for qa in qas:
        idx = qa[0]
        tar_par = contexts[idx]
        sentences = span_list(tar_par)
        for i, sent in enumerate(sentences):
            if qa[2] in sent.text:
                tar_sen = i
        item = (idx, qa[1], qa[2], tar_sen)
        qas_idx.append(item)
    return qas_idx


def main():
    contexts, qas = preprocess_unpickle("docs/squad contexts.pickle", "docs/squad qas.pickle")
    qas_idx = answer_text2idx(contexts, qas)
    pickle_all(qas_idx, "docs/squad idx_qas.pickle")


    # start = timer()
    # nlp = spacy_model()
    # preprocess_pickle(nlp, "docs/squad contexts.pickle", "docs/squad qas.pickle")
    # elapsed_time = timer()-start
    # print("to preprocess and pickle the docs, it took %s seconds" % elapsed_time)
    #
    # start = timer()
    # contexts, qas = preprocess_unpickle("docs/squad contexts.pickle", "docs/squad qas.pickle")
    # elapsed_time = timer()-start
    # print("to unpickle the docs and qas, it took %s seconds" % elapsed_time)

    start = timer()
    contexts, qas = preprocess_unpickle("docs/squad contexts.pickle", "docs/squad idx_qas.pickle")
    elapsed_time = timer()-start
    print("to unpickle the docs and qas, it took %s seconds" % elapsed_time)

    # This takes only 0.03 seconds
    qas = sorted(qas, key=lambda tup: tup[0])
    trimmed_qas = {qa for qa in qas if qa[0] < 100}
    trimmed_contexts = contexts[:100]

    # Processing 1165 questions with 100 context paragraphs took 4:23, averaging .22 seconds per qa pair
    # The accuracy is 0.71, meaning that the target paragraph was found in the top 10 results
    # The same processing with finding the most accurate paragraph, the accuracy was 0.31
    # [358, 143, 88, 53, 39, 29, 35, 30, 27, 23] - the top 10, and the distribution of the sentences
    rank_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    wrong = 0
    par_sent_correct = 0
    for qa in tqdm(trimmed_qas):
        # (paragraph_index, max_similarity_score, sentence_of_max)
        most_similar = rank_paragraphs(trimmed_contexts, qa[1], 10)
        xz = [ixx[0] for ixx in most_similar]
        try:
            va1 = xz.index(qa[0])
            rank_list[va1] += 1
            sen_idx = most_similar[va1][2]
            tar_idx = qa[3]
            if sen_idx == tar_idx:
                par_sent_correct += 1
        except ValueError:
            wrong += 1
    print(rank_list)
    print((len(trimmed_qas) - wrong)/len(trimmed_qas))
    print(wrong, len(trimmed_qas))
    print("correct sentence: ", par_sent_correct)

if __name__ == '__main__':
    main()
