from textblob import TextBlob
import spacy
import warnings


def spacy_model():
    """Returns a spacy "en_vectors_web_lg" model with an added sentencizer pipe"""
    nlp = spacy.load("en_vectors_web_lg")
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    return nlp


def textblob_sentencizer(text):
    """
    Returns a sentencized version of a text

    Parameters:
    text (str): A body of text to be sentencized

    Returns:
    str list: sentencized text as processed by TextBlob

    """
    blob = TextBlob(text)
    return [sent for sent in blob.sentences]


def spacy_sentence(text, nlp):
    """
    Returns a spacy doc object of text

    :param text: the text to be processed
    :param nlp: spacy model to process text
    :return: a spacy doc object of the parameter text
    """
    return nlp(text)


def span_list(doc):
    return [s for s in doc.sents if not s.text.isspace()]


def predict(similarity_list):
    """
    Predicts the target sentence by finding the sentence with the highest similarity value
    :param similarity_list: the list of similarity values
    :return: index of the span with the highest similarity value, the similarity value
    """
    index = similarity_list.index(max(similarity_list))
    return index, max(similarity_list)


def find_answer_sentence(context, question):
    """
    Predicts and returns the sentence containing the answer to the question
    :param context: spacy doc
    :param question: spacy doc
    :param nlp: spacy model, must be a vector model and contain a sentencizer
    :return: predict = index, similarity value
    """
    warnings.filterwarnings("error")
    similarity_list = []
    sentences = span_list(context)
    for yy, sent in enumerate(sentences):
        try:
            similarity_list.append(sent.similarity(question))
        except UserWarning:
            print(sent, question)
    return predict(similarity_list)


def find_answer_paragraph(paragraphs, question):
    """
    predicts which paragraph contains the answer to the question from an input of list of paragraphs
    :param paragraphs: list of strings, with each item in the list being paragraphs
    :param question: string
    :param nlp: a spacy model, must be a vector model with a sentencizer pipe
    :return: the index of the best paragraph from the list of paragraphs, the index of the best sentence from the paragraph, the string of the sentence, and the similarity value
    """
    highest_similarity = -1
    best_paragraph = -1
    sentence_id = -1
    for i, paragraph in enumerate(paragraphs):
        idx, max_similarity = find_answer_sentence(paragraph, question)
        if max_similarity > highest_similarity:
            highest_similarity = max_similarity
            sentence_id = idx
            best_paragraph = i
    return best_paragraph, highest_similarity, sentence_id


def rank_paragraphs(articles, question, max_length):
    most_similar = [] # set some value

    # most similar is a list of (paragraph_id, max_similarity)

    lowest_score = 1
    for i, paragraph in enumerate(articles):
        idx, max_similarity = find_answer_sentence(paragraph, question)
        # (paragraph index, max similarity score, sentence of the max similarity index)
        item = (i, max_similarity, idx)
        most_similar.append(item)
        if max_similarity < lowest_score:
            lowest_score = max_similarity
            if len(most_similar) <= max_length:
                continue
        if len(most_similar) > max_length:
            most_similar = sorted(most_similar, key=lambda tup: tup[1], reverse=True)
            del most_similar[-1]

    return most_similar


def main():
    return


if __name__ == '__main__':
    main()
