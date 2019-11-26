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
    spam = []
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
                if int(item[1]) == 1:
                    spam.append(string)
                right += 1
            else:
                wrong += 1
                wrong_items.append([item[0], item[1]])
    print(f'right: {right}, wrong: {wrong}')
    if type != "maxent":
        with open("spam.txt", "w") as f:
            for comment in spam:
                f.write("{}\n".format(comment))


def load_goodwords(filename):
    goodWords = []
    with open(filename, "r") as f:
        for line in f:
            goodWords.append(line.strip("\n"))
    return goodWords

def load_spam():
    spam = []
    with open("spam.txt", "r") as f:
        for line in f:
            spam.append(line.strip("\n"))
    return spam

def trick_bayes(classifier):
    WORD_LIMIT = 200
    goodWords = load_goodwords('Bayes_Data/goodwords.txt')
    spam = load_spam()


    avg = 0
    flag = False
    import random 
    for comment in spam:
        for i in range(WORD_LIMIT):
            comment += " {}".format(random.choice(goodWords))
            if classifier.predict(comment) == False:
                avg += i
                break
            if i == WORD_LIMIT - 1:
                avg += WORD_LIMIT
    print(avg / len(spam))


def trick_maxent(classifier):
    WORD_LIMIT = 200
    goodWords = load_goodwords('Maxent_Data/goodwords.txt')
    spam = load_spam()
    spam = [line.split(" ")[1] for line in spam]



    avg = 0
    flag = False
    import random 
    for comment in spam:
        for i in range(0, WORD_LIMIT):
            comment += " {}".format(random.choice(goodWords))
            if classifier.classify(list_to_dict(comment)) == str(0):
                avg += i
                break
            if i == WORD_LIMIT - 1:
                avg += WORD_LIMIT
    print(avg/len(spam))


'''
Active Attacks
'''

def find_witness(spam, legit, classifier):
    length = min([len(spam), len(legit)])

    curr = spam
    i = 0
    flag = False
    prev = curr
    if type(classifier) == NaiveClassifier:
        while classifier.predict("".join(curr)) != True:
            prev = curr
            for j in range(len(curr)):
                if curr[j] not in spam:
                    del curr[j]
                    flag = True
                    break 
            if not flag:
                for j in range(len(spam)):
                    if spam[j] not in curr:
                        curr.append(j)
                        break 
            flag = False
            i += 1
            if i == length:
                return curr, prev
    elif type(classifier) == nltk.classify.maxent.MaxentClassifier:
        while classifier.classify(list_to_dict(curr)) != str(1):
            prev = curr
            for j in range(len(curr)):
                if curr[j] not in spam:
                    del curr[j]
                    flag = True
                    break 
            if not flag:
                for j in range(len(spam)):
                    if spam[j] not in curr:
                        curr.append(j)
                        break 
            flag = False
            i += 1
            if i == length:
                return curr, prev
    else:
        while classifier.classify( "".join(curr) ) != 0:
            prev = curr
            for j in range(len(curr)):
                if curr[j] not in spam:
                    del curr[j]
                    flag = True
                    break 
            if not flag:
                for j in range(len(spam)):
                    if spam[j] not in curr:
                        curr.append(j)
                        break 
            flag = False
            i += 1
            if i == length:
                return curr, prev
            
    return (curr, prev)


def first_n_words(spam, legit, classifier, wordlist):
    L = []
    to_spam, _ = find_witness(spam, legit, classifier)
    for word in wordlist:
        to_spam.append(word)
        if type(classifier) == NaiveClassifier:
            if classifier.predict("".join(to_spam)) == False:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        elif type(classifier) == nltk.classify.maxent.MaxentClassifier:
            if classifier.classify(list_to_dict(spam)) == str(0):
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        else:
            if classifier.classify("".join(to_spam)) == 0:
                L.append(word)
                to_spam = to_spam[:len(to_spam)-1]
        if len(L) == 20:
            break 
        
    return L

def get_word_list():
    wordlist = []
    with open("unique.txt", "r") as f:
        for line in f:
            wordlist.append(line.strip("\n"))
    return wordlist

def get_spam_and_legit():
    spamlist, legitlist = [], []
    with open("datasets/Psy.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == "COMMENT_ID":
                continue
            content = re.sub(r'[^\w\s]', '', row[3]).lower().split(" ")

            if int(row[4]) == 1:
                spamlist.append(content)
            else:
                legitlist.append(content)
    return (spamlist, legitlist)

def run_first_n_words(classifier):
    '''
    results = lovely girl talk to me xxx i always end up coming back this songbr my sister just received over 6500
    '''
    spamlist, legitlist = get_spam_and_legit()
    wordlist = get_word_list()
    length = min([len(spamlist), len(legitlist)])
    n_words = set()
    for i in range(length):
        n_words.add(" ".join(first_n_words(spamlist[i], legitlist[i], classifier, wordlist)))
    return list(n_words)


def best_n_words(spam, legit, classifier, wordlist):
    pass
    M_p, M_n = find_witness(spam, legit, classifier)
    S = M_n
    L = M_p
    best = []
    for word in S:
        small = S
        large = L
        if len(large) + len(best) < 15:
            for word in large:
                best.append(word)
            L = []
        else:
            for word in small:
                L.remove(word)
    return best




def run_best_n_words(classifier):
    spamlist, legitlist = get_spam_and_legit()
    wordlist = get_word_list()
    length = min(len(spamlist), len(legitlist))
    n_words = set()
    for i in range(length):
        n_words.add(" ".join(best_n_words(spamlist[i], legitlist[i], classifier, wordlist)))
    return list(n_words)

def first_n_attack(classifier):
    WORD_LIMIT = 200
    goodWords = run_first_n_words(classifier)
    spam = load_spam()

    if type(classifier) == NaiveClassifier:
        print("Naive Bayes")
    else:
        print("Maxent")

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
            else:
                if classifier.classify(list_to_dict(comment.split(" "))) == str(0):
                    avg += i
                    break 
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
    print(avg / len(spam))

def best_n_attack(classifier):
    WORD_LIMIT = 200
    goodWords = run_best_n_words(classifier)
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
            else:
                if classifier.classify(list_to_dict(comment.split(" "))) == str(0):
                    avg += i
                    break 
                if i == WORD_LIMIT - 1:
                    avg += WORD_LIMIT
    print(avg / len(spam))

def main():
    bayes = load_bayes()
    maxent = load_maxent()

    # First N attacks
    first_n_attack(bayes)
    first_n_attack(maxent)

if __name__ == "__main__":
    main()