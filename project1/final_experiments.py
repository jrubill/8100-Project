from experiments import load_bayes, load_maxent, run_first_n_words, load_spam
from bayes import NaiveClassifier, Word
from multiple import RFClassifier, SVM, Voting, MLP
import multiple
import nltk
import re
import csv


def load_SVM():
    clf = multiple.SVM()
    clf.load_data('datasets/training.csv')
    clf.train()
    return clf

def load_RF():
    clf = RFClassifier()
    #clf.load_data('datasets/KatyPerry.csv')
    clf.load_data('datasets/training.csv')
    clf.train()
    return clf

def load_VCLF():
    clf = Voting()
    clf.load_data('datasets/training.csv')
    clf.train()
    return clf

def load_MLP():
    clf = MLP()
    clf.load_data('datasets/training.csv')
    clf.train()
    return clf

def first_n_words(spam, legit, classifier, wordlist):
    L = []
    to_spam, _ = find_witness(spam, legit, classifier)
    for word in wordlist:
        to_spam.append(word)
        if type(classifier) == NaiveClassifier:
            if classifier.predict("".join(to_spam)) == False:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        elif type(classifier) == mltk.classify.maxent.MaxentClassifier:
            if classifier.classify(list_to_dict(spam)) == str(0):
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        else:
            if classifier.classify(spam) == 0:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        if len(L) == 20:
            break 
    return L


def first_n_attack(classifier):
    WORD_LIMIT = 200
    goodWords = run_first_n_words(classifier)
    spam = load_spam()

    avg = 0
    flag = False 
    import random
    for comment in spam:
        for i in range(WORD_LIMIT):
            comment += " {}".format(random.choice(goodWords))
            if type(classifier) == NaiveClassifier:
                if classifier.predict(comment) == False:
                    avg += i
                    break
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
            elif type(classifier) == nltk.classify.maxent.MaxentClassifier:
                if classifier.classify(list_to_dict(comment.split(" "))) == str(0):
                    avg += i
                    break 
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
            else:
                if classifier.classify(comment) == 0:
                    avg += i
                    break 
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
    print(type(classifier))
    print(avg / len(spam))

def test_classifier(classifier, filename):
    wrong, correct = 0, 0
    with open(filename, "r", encoding='utf8') as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue
            data = re.sub(r'[^\w\s]', '', row[3]).lower()
            prediction = classifier.classify(data)
            actual = int(row[4]) == 1
            if prediction and actual == 1 or not prediction and actual == 0:
                correct += 1
            else:
                wrong += 1
    print(f'wrong: {wrong}, right: {correct}, avg={correct/(correct+wrong)}')


def time_this(function, classifier):
    from datetime import datetime 
    before = datetime.now()
    function(classifier, 'datasets/KatyPerry.csv')
    after = datetime.now()
    return after - before

if __name__ == "__main__":
    bayes = load_bayes()
    maxent = load_maxent()
    RF = load_RF()
    SVM = load_SVM()
    vclf = load_VCLF()
    mlp = load_MLP()

    #first_n_attack(vclf)
    #first_n_attack(bayes)
    #first_n_attack(maxent)
    #first_n_attack(SVM)
    #first_n_attack(RF)
    '''
    for i in range(5):
        test_classifier(vclf, 'datasets/LMFAO.csv')
        test_classifier(RF, 'datasets/LMFAO.csv')
    '''
    classifiers = [RF, SVM, vclf, mlp]
    timers = [0]*4
    for i in range(10):
        for j in range(len(classifiers)):
            timers[j] += time_this(test_classifier, classifiers[j]).total_seconds()
    for i in range(4):
        timers[i] /= 10
    
    print(timers)


    '''
    for i in range(5):
        test_classifier(load_SVM(), 'datasets/KatyPerry.csv')
    '''
    '''
    params = [load_VCLF() for i in range(5)]
    for item in params:
        first_n_attack(item)
    '''
    '''
    for item in params:
        first_n_attack(item)
        #from multiprocessing import Pool
    '''