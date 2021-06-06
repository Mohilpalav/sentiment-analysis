import math, os, pickle, re, random, string
from collections import defaultdict

class Bayes_Classifier:

    def __init__(self, trainDir = "reviews/movie_and_product_reviews/"):
        '''This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
        cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
        the system will proceed through training.  After running this method, the classifier 
        is ready to classify input text.'''

        self.Path = trainDir
        self.training_data = []
        self.testing_data = []

        self.punctuation = [char for char in string.punctuation]
        # self.stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        self.positive_document = defaultdict(self.on_key_error)
        self.negative_document = defaultdict(self.on_key_error)
        self.model_information = {}
        
        if os.path.exists("positive_document") and os.path.exists("negative_document") and os.path.exists("model_information"):
            self.positive_document = self.load("positive_document")
            self.negative_document = self.load("negative_document")
            self.model_information = self.load("model_information")

        else:
            self.train()
            self.positive_document = self.load("positive_document")
            self.negative_document = self.load("negative_document")
            self.model_information = self.load("model_information")
            

    def train(self):   
        '''Trains the Naive Bayes Sentiment Classifier.'''

        IFileList = []
        for fFileObj in os.walk(self.Path):  
            IFileList = fFileObj[2]
            break

        random.shuffle(IFileList)
        size = len(IFileList)
        split = int(0.2*size)

        self.testing_data = IFileList[0:split]
        self.training_data = IFileList[split:]

        positive_text_count = 0
        negative_text_count = 0

        for filename in self.training_data:
            rating = filename.split('-')[1]

            if rating == '1':
                negative_text_count += 1
            elif rating == '5':
                positive_text_count += 1

        length = min(negative_text_count, positive_text_count)
        positive_text_count = length
        negative_text_count = length
        
        for filename in self.training_data:

            rating = filename.split('-')[1]
            if(positive_text_count <= 0 and negative_text_count <= 0):
                break

            if rating in ['1', '5']:

                content = self.loadFile(self.Path + filename)
                words = self.tokenize(content)  
                pairs = [(words[i] + " "+ words[i+1]) for i in range(0, len(words)-1)]

                if rating == '1':
                    if(negative_text_count > 0):
                        negative_text_count -= 1
                        for pair in pairs:
                            self.negative_document[pair] += 1
                    
                elif rating == '5':
                    if(positive_text_count > 0):
                        positive_text_count -= 1
                        for pair in pairs:
                            self.positive_document[pair] += 1

        self.model_information = {
            "positive_count": length,
            "negative_count": length,
            "testing_files": self.testing_data
        }

        self.save(self.positive_document, "positive_document")
        self.save(self.negative_document, "negative_document")
        self.save(self.model_information, "model_information")
        
        
    def classify(self, sText):
        '''Given a target string sText, this function returns the most likely document
        class to which the target string belongs. This function should return one of three
        strings: "positive", "negative" or "neutral".
        '''

        positive_count = self.model_information["positive_count"]
        negative_count = self.model_information["negative_count"]

        positive_product = 0 
        negative_product = 0

        positive_probability = math.log(positive_count / (positive_count + negative_count))
        negative_probability = math.log(negative_count / (positive_count + negative_count))

        positive_words = sum(self.positive_document.values())
        negative_words = sum(self.negative_document.values())

        words = self.filter_tokens(self.tokenize(sText))
        pairs = [(words[i] + " "+ words[i+1]) for i in range(0, len(words)-1)]
        
        for pair in pairs:
            positive_product += math.log((self.positive_document[pair]+1) / (positive_words + 1 * len(self.positive_document)))
            negative_product += math.log((self.negative_document[pair]+1) / (negative_words + 1 * len(self.negative_document)))

        positive_product += positive_probability
        negative_product += negative_probability
        
        if abs(positive_product - negative_product) < 0.5:
            return "neutral"

        if positive_product > negative_product:
            return "positive"
        elif positive_product < negative_product:
            return "negative"
        

    def loadFile(self, sFilename):
        '''Given a file name, return the contents of the file as a string.'''

        f = open(sFilename, "r", encoding="utf8")
        sTxt = f.read()
        f.close()
        return sTxt

    def save(self, dObj, sFilename):
        '''Given an object and a file name, write the object to the file using pickle.'''

        f = open(sFilename, "wb")
        p = pickle.Pickler(f)
        p.dump(dObj)
        f.close()

    def load(self, sFilename):
        '''Given a file name, load and return the object stored in the file.'''

        f = open(sFilename, "rb")
        u = pickle.Unpickler(f)
        dObj = u.load()
        f.close()
        return dObj

    def tokenize(self, sText):
        '''Given a string of text sText, returns a list of the individual tokens that
        occur in that string (in order).'''

        lTokens = []
        sToken = ""
        for c in sText:
            if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\'" or c == "_" or c == '-':
                sToken += c
            else:
                if sToken != "":
                    lTokens.append(sToken)
                    sToken = ""
                if c.strip() != "":
                    lTokens.append(str(c.strip()))

        if sToken != "":
            lTokens.append(sToken)

        return lTokens

    def filter_tokens(self, tokens):
        '''Given a list of tokens, returns a list of preprocessed tokens'''

        filtered_tokens = []
        for token in tokens:
            # if(token not in self.punctuation and token not in self.stop_words):
            #     filtered_tokens.append(token)
            if(token not in self.punctuation):
                filtered_tokens.append(token)

        return filtered_tokens

    def on_key_error(self):
        '''Returns 0 on Dictionary Key Error'''
        return 0

   
