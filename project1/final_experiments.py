from experiments import load_bayes, load_maxent, run_first_n_words, load_spam, get_spam_and_legit, get_word_list, find_witness
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


def run_first_n_words(classifier, n):
    spamlist, legitlist = get_spam_and_legit()
    wordlist = get_word_list()
    length = min([len(spamlist), len(legitlist)])
    n_words = set()
    for i in range(length):
        n_words.add(" ".join(first_n_words(spamlist[i], legitlist[i], classifier, wordlist, n)))
    return list(n_words)


def first_n_words(spam, legit, classifier, wordlist,n):
    L = []
    to_spam, _ = find_witness(spam, legit, classifier)
    for word in wordlist:
        to_spam.append(word)
        if type(classifier) == NaiveClassifier:
            if classifier.predict(" ".join(to_spam)) == False:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        elif type(classifier) == nltk.classify.maxent.MaxentClassifier:
            if classifier.classify(list_to_dict(spam)) == str(0):
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        else:
            if classifier.classify(" ".join(spam)) == 0:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        if len(L) == n:
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



def best_n_attack(classifier):
    WORD_LIMIT = 200
    spam = load_spam()
    bestwords = find_best_n_words(classifier, 20)
    avg = 0

    for comment in spam:
        for i in range(len(bestwords)):
            comment += " {}".format(bestwords[i])
            if type(classifier) == NaiveClassifier:
                if classifier.predict(comment) == False:
                    avg += i
                    break
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
            elif type(classifier) ==  nltk.classify.maxent.MaxentClassifier:
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
    return(avg / len(spam))



def find_best_n_words(classifier, n):
    goodwords = run_first_n_words(classifier, n)
    worddict = {}
    spam = load_spam()

    for goodword in goodwords:
        for word in goodword.split(" "):
            if word not in worddict:
                worddict[word] = 0

            for comment in spam:
                comment_modified = comment + " {}".format(word)
                if type(classifier) == NaiveClassifier:
                    if not classifier.predict(comment_modified):
                        worddict[word] += 1
                elif type(classifier) == nltk.classify.maxent.MaxentClassifier:
                    if classifier.classify(list_to_dict(comment_modified.split(" "))) == str(0):
                        worddict[word] += 1
                else:
                    if classifier.classify(comment_modified) == 0:
                        worddict[word] += 1
                

    #print("worddict: " + str(worddict))
    bestwords = [word for word in sorted(worddict, key=worddict.get, reverse=True)]
    return bestwords



if __name__ == "__main__":
    bayes = load_bayes()
    #maxent = load_maxent()
    RF = load_RF()
    SVM = load_SVM()
    vclf = load_VCLF()
    mlp = load_MLP()

    #first_n_attack(vclf)
    #first_n_attack(bayes)
    #first_n_attack(maxent)
    #first_n_attack(SVM)
    #first_n_attack(RF)
    best_n_attack(SVM)
    '''
    for i in range(5):
        test_classifier(RF, 'datasets/KatyPerry.csv')
    
    classifiers = [RF, SVM, vclf, mlp]
    for item in classifiers:
        first_n_attack(item)
    '''
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