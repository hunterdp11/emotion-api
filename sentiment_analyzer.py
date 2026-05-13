import re
import math
import random
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure stopwords are available
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

stemmer = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [stemmer.stem(w) for w in words if w not in stop_words]
    return " ".join(words)

def vectorize(text, vocab):
    words = text.split()
    return [words.count(w) for w in vocab]

def softmax(z):
    exp_z = [math.exp(i - max(z)) for i in z]
    s = sum(exp_z)
    return [i / s for i in exp_z]

def relu(x):
    return [max(0, i) for i in x]

# Classes required for unpickling the model
class NaiveBayes:
    def __init__(self):
        self.class_word_counts = {}
        self.class_counts = {}
        self.vocab_size = 0

    def predict(self, x):
        scores = {}
        total_docs = sum(self.class_counts.values())
        for label in self.class_counts:
            log_prob = math.log(self.class_counts[label] / total_docs)
            total_words = sum(self.class_word_counts[label])
            for i in range(len(x)):
                prob = (self.class_word_counts[label][i] + 1) / (total_words + self.vocab_size)
                log_prob += x[i] * math.log(prob)
            scores[label] = log_prob
        return max(scores, key=scores.get)

class LogisticRegressionScratch:
    def __init__(self, lr=0.05, epochs=300):
        self.lr = lr
        self.epochs = epochs
        self.W = None
        self.b = None

    def predict(self, X):
        return [
            max(range(3), key=lambda c: sum(w*xi for w, xi in zip(self.W[c], x)) + self.b[c])
            for x in X
        ]

class NeuralNetworkScratch:
    def __init__(self, input_size, hidden=16, lr=0.01, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.W1 = None
        self.b1 = None
        self.W2 = None
        self.b2 = None

    def predict(self, X):
        preds = []
        for x in X:
            a1 = relu([sum(w*xi for w, xi in zip(ws, x)) + b for ws, b in zip(self.W1, self.b1)])
            z2 = [sum(w*ai for w, ai in zip(ws, a1)) + b for ws, b in zip(self.W2, self.b2)]
            preds.append(max(range(3), key=lambda i: z2[i]))
        return preds
