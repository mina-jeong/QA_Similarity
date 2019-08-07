from timeit import default_timer as timer
from similarity import spacy_model, spacy_sentence, rank_paragraphs, span_list
from pickle_docs import *

# the main is not for SQUAD it is for BBC news
def main():
    """
    This function simulates what would happen if a user asked questions regarding the BBC articles, in a pickle file
    :return:
    """
    start = timer()
    print("loading spacy model...")
    nlp = spacy_model()
    text_path = r'C:\Users\mjeong\PycharmProjects\nlp-comprehension\texts\bbc'
    docs = []
    for folder in os.scandir(text_path):
        print(folder)
        if folder.is_dir():
            docs = scan_files(folder, docs, nlp)
    pickle_all(docs, "docs/bbc_docs.pickle")

    docs = unpickle_all("docs/bbc_docs.pickle")
    run = True
    end = timer() - start
    print(end)
    while run:
        question_text = input("Enter a question: ")
        if question_text == "exit":
            run = False
            continue
        question = spacy_sentence(question_text, nlp)
        most_similar = rank_paragraphs(docs, question, 5)
        print(most_similar)
        for index in range(len(most_similar)):
            i = most_similar[index][0]
            j = most_similar[index][2]
            print(" ")
            print("answer sentence")
            print("----------------------------------")
            print(span_list(docs[i])[j])
            print(" ")


if __name__ == '__main__':
    main()
