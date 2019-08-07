import pickle
from similarity import spacy_sentence
import json
import os


def import_text_to_docs(path, nlp):
    f = open(path).read()
    docs = spacy_sentence(f, nlp)
    return docs


def pickle_all(docs, path):
    # path = '/Users/mjeong/PycharmProjects/nlp-comprehension/pickle/docs.pkl'
    with open(path, "wb") as f:
        pickle.dump(docs, f)


def unpickle_all(path):
    with open(path, "rb") as f:
        doc = pickle.load(f)
    return doc


def json_context_extract(filename, nlp):
    contexts = []
    with open(filename, encoding='utf-8') as data_file:
        data = json.load(data_file)
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context = paragraph['context']
            context = spacy_sentence(context, nlp)
            contexts.append(context)
    return contexts


def json_qa_extract(filename, nlp):
    qas = []
    context_index = -1
    with open(filename, encoding='utf-8') as data_file:
        data = json.load(data_file)
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context_index += 1
            for qa in paragraph['qas']:
                is_impossible = qa['is_impossible']
                if not is_impossible:
                    question = qa['question']
                    question = spacy_sentence(question, nlp)
                    answer = qa['answers'][0]['text']
                    qas.append((context_index, question, answer))
    print(len(qas))
    print(qas)
    return qas


def json_context_qas_extract(filename, nlp):
    qas = []
    contexts = []
    context_index = -1
    with open(filename, encoding='utf-8') as data_file:
        data = json.load(data_file)
    for item in data['data']:
        for paragraph in item['paragraphs']:
            context_index += 1
            context = paragraph['context']
            context = spacy_sentence(context, nlp)
            contexts.append(context)
            for qa in paragraph['qas']:
                is_impossible = qa['is_impossible']
                if not is_impossible:
                    question = qa['question']
                    question = spacy_sentence(question, nlp)
                    answer = qa['answers'][0]['text']
                    qas.append((context_index, question, answer))
    print(len(qas))
    print(len(contexts))
    return contexts, qas


def scan_files(path, docs, nlp):
    print(path)
    for file in os.scandir(path):
        docs.append(import_text_to_docs(file, nlp))
    return docs
