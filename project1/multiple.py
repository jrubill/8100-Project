# Multiple classifiers to be used at once
import csv
import re
import math
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
# CNN
class CNN(object):
    def __init__(self):
        pass

class RFClassifier():
    def __init__(self):
        self.vect = CountVectorizer(max_features=1500)
    
    def load_data(self, filename):
        self.y = []
        self.x = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'COMMENT_ID':
                    continue
                content = re.sub(r'[^\w\s]', '', row[3]).lower()
                content = content.split(" ")
                self.y.append(int(row[4]))
                self.x.append( ' '.join(content))
        self.x = self.vect.fit_transform(self.x).toarray()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
    
    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2)

        self.classifier = RandomForestClassifier(n_estimators=15, criterion='entropy', n_jobs=1)
        self.classifier.fit(x_train, y_train)
        predRF = self.classifier.predict(x_test)
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        '''
        print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
        print('Precision score: {}'.format(precision_score(y_test, predRF)))
        print('Recall score: {}'.format(recall_score(y_test, predRF)))
        print('F1 score: {}'.format(f1_score(y_test, predRF)))
        '''

    def classify(self, text):
        text = self.vect.transform([text])
        return self.classifier.predict(text)


class SVM():
    def __init__(self):
        self.vect = CountVectorizer(max_features=1500)
 
    def load_data(self, filename):
        self.x = []
        self.y = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'COMMENT_ID':
                    continue
                content = re.sub(r'[^\w\s]', '', row[3]).lower()
                content = content.split(" ")
                self.y.append(int(row[4]))
                self.x.append( ' '.join(content))
        self.x = self.vect.fit_transform(self.x).toarray()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
    
    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y,test_size = 0.1)
        #x_train, y_train = self.x, self.y
        # self.classifier = svm.SVC(C=1.0, kernel='linear')
        self.classifier = svm.SVC(kernel='linear')
        self.classifier.fit(x_train, y_train)
        #predRF = self.classifier.predict(x_test)
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        '''
        print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
        print('Precision score: {}'.format(precision_score(y_test, predRF)))
        print('Recall score: {}'.format(recall_score(y_test, predRF)))
        print('F1 score: {}'.format(f1_score(y_test, predRF)))
        '''
    def classify(self, text):
        text = self.vect.transform([text]).todense()
        return self.classifier.predict(text)

class Voting():
    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.naive_bayes import GaussianNB
        from sklearn.ensemble import VotingClassifier
        self.classifiers=[('mlp',MLPClassifier()), ('rf', RandomForestClassifier(n_estimators=5)), ('svm', svm.SVC(kernel='linear'))]
        self.classifier = VotingClassifier(estimators=self.classifiers, voting='hard', n_jobs=-1)
        self.vect = CountVectorizer(max_features=1500)
    
    def load_data(self, filename):
        self.x = []
        self.y = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'COMMENT_ID':
                    continue
                content = re.sub(r'[^\w\s]', '', row[3]).lower()
                content = content.split(" ")
                self.y.append(int(row[4]))
                self.x.append( ' '.join(content))
        self.x = self.vect.fit_transform(self.x).toarray()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
    
    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2)

        self.classifier.fit(x_train, y_train)
        predRF = self.classifier.predict(x_test)
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        '''
        print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
        print('Precision score: {}'.format(precision_score(y_test, predRF)))
        print('Recall score: {}'.format(recall_score(y_test, predRF)))
        print('F1 score: {}'.format(f1_score(y_test, predRF)))
        '''
    def classify(self, text):
        text = self.vect.transform([text]).todense()
        return self.classifier.predict(text)

class MLP():
    def __init__(self):
        self.classifier = MLPClassifier()
        self.vect = CountVectorizer(max_features=1500)
    
    def load_data(self, filename):
        self.x = []
        self.y = []
        with open(filename, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                if row[0] == 'COMMENT_ID':
                    continue
                content = re.sub(r'[^\w\s]', '', row[3]).lower()
                content = content.split(" ")
                self.y.append(int(row[4]))
                self.x.append( ' '.join(content))
        self.x = self.vect.fit_transform(self.x).toarray()
        le = LabelEncoder()
        self.y = le.fit_transform(self.y)
    
    def train(self):
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size = 0.2)

        self.classifier.fit(x_train, y_train)
        predRF = self.classifier.predict(x_test)
        from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
        '''
        print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
        print('Precision score: {}'.format(precision_score(y_test, predRF)))
        print('Recall score: {}'.format(recall_score(y_test, predRF)))
        print('F1 score: {}'.format(f1_score(y_test, predRF)))
        '''
    def classify(self, text):
        text = self.vect.transform([text]).todense()
        return self.classifier.predict(text)

if __name__ == "__main__":
    pass
    