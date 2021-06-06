# reviews-sentiment-analysis

Sentiment analysis of movie and actor reviews using N-gram model and Naïve-Bayes Classifier in python.

We train the model using positive and negative reviews using an unigram model and use the model to classify whether a review is positive, negative or neutral using the Naïve-Bayes Classifier. We evaluate the model using macro averaging. The second model uses bigrams and a few preprocessing techniques to improve the overall accuracy.

### Execution Steps:

1. Clone the repository
2. Alter the filename in the import statement in evaluate.py to the file you want to evaluate.
3. Alter the trainDir variable in evaluate.py and provide the training folder path.
4. Provide your testing folder path through the command line as follows:
    '''
    python3 evaluate.py "TEST_PATH"
    '''
 
