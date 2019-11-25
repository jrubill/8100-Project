# Multiple classifiers to be used at once
import csv
import re
import math

# CNN
class CNN(object):
    def __init__(self):
        pass

# RNN

def features(words):
    return dict([(word, True) for word in words])

def load_data(filename):
    illegalWords = set(['katy', 'perry', 'katyperry', 'psy', 'shakira', 'eminem'])
    training_set = []
    with open(filename, 'r', encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue
            content = re.sub(r'[^\w\s]', '', row[3]).lower()

            content = set(content.split(" "))
            for word in illegalWords:
                if word in content:
                    content.remove(word)
            content = list(content)
            training_set.append([content, row[4]])
    return training_set
    


def list_to_dict(words):
    return dict([(word, True) for word in words])

def isSpam(val):
    if val == 1:
        return "Spam"
    return "Not Spam"

if __name__ == "__main__":
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.ensemble import RandomForestClassifier

    vectorizer = CountVectorizer(max_features=1500)
    y = []
    words = []
    with open("datasets/KatyPerry.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue
            content = re.sub(r'[^\w\s]', '', row[3]).lower()
            content = content.split(" ")
            y.append(int(row[4]))
            words.append( ' '.join(content))

    x = vectorizer.fit_transform(words).toarray()
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

    classifier = RandomForestClassifier(n_estimators=15, criterion='entropy')
    classifier.fit(x_train, y_train)
    predRF = classifier.predict(x_test)
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    print('Accuracy score: {}'.format(accuracy_score(y_test, predRF)))
    print('Precision score: {}'.format(precision_score(y_test, predRF)))
    print('Recall score: {}'.format(recall_score(y_test, predRF)))
    print('F1 score: {}'.format(f1_score(y_test, predRF)))

    x = vectorizer.transform(["hello"])#.toarray()
    print(classifier.predict(x))
    