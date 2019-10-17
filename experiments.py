import sys
import pickle 
from bayes import NaiveClassifier, Word
from n_maxent import list_to_dict
import nltk
import csv 
import math 
import re

def load_maxent():
    classifier = nltk.MaxentClassifier
    try:
        with open("maxent.pickle", "rb") as f:
            classifier = pickle.load(f)
            return classifier
    except IOError:
        print("Pickle for maxent not found. Exiting. . .")

def load_bayes():
    classifier = NaiveClassifier
    try:
        with open("bayes.pickle", "rb") as f:
            classifier = pickle.load(f)
            return classifier
    except IOError:
        print("Pickle for bayes not found. Exiting. . .")
        sys.exit(1)


def test(filename, classifier, type="maxent"):
    testing_set = []
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue 
            content = re.sub(r'[^\w\s]', '', row[3]).lower()
            content = content.split(" ")
            testing_set.append([content, row[4]])

    right = 0
    wrong = 0
    wrong_items = []
    for item in testing_set:
        if type == "maxent":
            if classifier.classify(list_to_dict(item[0])) == item[1]:
                right += 1
            else: 
                wrong += 1
                wrong_items.append([item[0], item[1]])
        else:
            string = ""
            for word in item[0]:
                string += "{} ".format(word)
            if classifier.predict(string) == int(item[1]):
                right += 1
            else:
                wrong += 1
                wrong_items.append([item[0], item[1]])
    print(f'# right {right} # wrong {wrong}')
    #print(wrong_items)

if __name__ == "__main__":
    bayes = load_bayes()
    maxent = load_maxent()

    test('datasets/Shakira.csv', bayes, "bayes")
    test('datasets/Shakira.csv', maxent)