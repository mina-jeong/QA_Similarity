# QA_Similarity
This is my code for finding the answer to a question within a context by using vector similarity.


## This is code for similarity comparison
Similarity.py contains the code for a spacy-processed paragraph to sentence similarity comparison.


It can also return which sentence in the paragraph is the most similar to the sentence.

There is a function in Similarity.py that can return a list of articles that is the most similar to the question.


This is calculated by looking at each paragraph, then searching for a target sentence within the paragraph.
The paragraph's target sentence's similarity value is then compared to the other paragraphs' target sentences.


What is returned as the articles that are the most similar to the question are the articles
 containing the sentences that are the most similar to the question
