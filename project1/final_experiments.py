from experiments import load_bayes, load_maxent, run_first_n_words, load_spam
from bayes import NaiveClassifier, Word
from multiple import RFClassifier, SVM, Voting
import nltk


def load_SVM():
    clf = SVM()
    clf.load_data('datasets/KatyPerry.csv')
    clf.train()
    return clf

def load_RF():
    clf = RFClassifier()
    clf.load_data('datasets/KatyPerry.csv')
    clf.train()
    return clf

def load_VCLF():
    clf = Voting()
    clf.load_data('datasets/KatyPerry.csv')
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

if __name__ == "__main__":
    bayes = load_bayes()
    maxent = load_maxent()
    RF = load_RF()
    SVM = load_SVM()
    vclf = load_VCLF()

    first_n_attack(vclf)
    #first_n_attack(bayes)
    #first_n_attack(maxent)
    #first_n_attack(SVM)
    #first_n_attack(RF)
    '''
    from multiprocessing import Pool
    with Pool(2) as p:
        p.map(first_n_attack, [RF, SVM])
    '''