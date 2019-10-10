import csv
import re
import math

DIR = 'datasets/'
DATASET = 'Eminem.csv'


''' 
    If something is spam, we denote it as positive. 
    If it is not positive, we define it as negative.
'''

class NaiveClassifier:
    def __init__(self):
        pass

    def load_data(self, filename):
        illegalWords = set(['katy', 'perry', 'katyperry', 'psy', 'shakira', 'eminem'])
        self.wordDict = {}
        self.neg_num, self.pos_num  = 0, 0
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'COMMENT_ID':
                    continue
                content = re.sub(r'[^\w\s]', '', row[3]).lower()
                content = content.split(" ")
                for word in content:
                    if word in illegalWords:
                        continue
                    if not word in self.wordDict:
                        self.wordDict[word] = Word(word)
                    if int(row[4]) == 1: # its spam
                        self.wordDict[word].positive += 1
                        self.pos_num += 1
                    else:
                        self.wordDict[word].negative += 1
                        self.neg_num += 1

        self.total = len(self.wordDict.keys())
        #pos_num, neg_num = len(pos_set), len(neg_set)
        for word in self.wordDict:
            self.wordDict[word].positive = self.laplace(self.total, self.pos_num, self.wordDict[word].positive)
            self.wordDict[word].negative = self.laplace(self.total, self.neg_num, self.wordDict[word].negative)
        
    def predict(self, msg):
        data = re.sub(r'[^\w\s]', '', msg).lower().split(" ")
        pos, neg = 1, 1
        for word in data:
            if word in self.wordDict:
                pos *= self.wordDict[word].positive
                neg *= self.wordDict[word].negative
            else:
                pos *= self.laplace(self.total, self.pos_num, 0)
                neg *= self.laplace(self.total, self.neg_num, 0)

        if pos > neg:
            return True
        return False

    def laplace(self, total, unique, val):
        return (1 + val) / (total + unique)

class Word:
    def __init__(self, key):
        self.key = key
        self.negative = 0
        self.positive = 0
    
    def __hash__(self):
        return self.key
    
    def prob(self):
        pass

    def __eq__(self, rhs):
        return self.key == rhs.key

def test_classifer(classifier, filename):
    wrong, correct = 0, 0
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue
            data = re.sub(r'[^\w\s]', '', row[3]).lower()
            prediction = classifier.predict(data)
            actual = int(row[4]) == 1
            if prediction and actual == 1 or not prediction and actual == 0:
                correct += 1
            else:
                wrong += 1
    
    print(f'prediction accuracy: {correct/(wrong + correct)}')

if __name__ == "__main__":
    c = NaiveClassifier()
    c.load_data(DIR+DATASET)
    #c.load_data(DIR+"Shakira.csv")
    #c.load_data(DIR+'LMFAO.csv')
    #c.load_data(DIR+"Psy.csv")

    test_classifer(c, DIR+'KatyPerry.csv')
    