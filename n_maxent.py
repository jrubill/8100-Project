import nltk
import csv
import re
import math
import pickle

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

            content = content.split(" ")
            for i in range(len(content)):
                if i in illegalWords:
                    del content[i]
            training_set.append([content, row[4]])
    return training_set
    


def list_to_dict(words):
    return dict([(word, True) for word in words])

def isSpam(val):
    if val == 1:
        return "Spam"
    return "Not Spam"

def main():
    val = ""
    while val != "quit":
        if val:
            print(classifier.classify(list_to_dict(val.split(" "))))
        val = input("please enter text:")

def test(filename,  classifier):
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
        if classifier.classify(list_to_dict(item[0])) == item[1]:
            right += 1
        else: 
            wrong += 1
            wrong_items.append([item[0], item[1]])
    print(f'# right {right} # wrong {wrong}')



if __name__ == "__main__":
    classifier = ""
    try:
        with open("maxent.pickle", "rb") as f:
            classifier = pickle.load(f)
    except IOError:
        training_set = [(list_to_dict(item[0]), item[1]) for item in load_data('datasets/Eminem.csv')]
        training_set += [(list_to_dict(item[0]), item[1]) for item in load_data('datasets/Shakira.csv')]

        iterations = 5
        classifier = nltk.MaxentClassifier.train(training_set, max_iter = iterations, gaussian_prior_sigma=0.01) # gaussian_prior_sigma=0.1
        
        with open("maxent.pickle", "wb") as f:
            pickle.dump(classifier, f)
    '''
    classifier.show_most_informative_features(10)

    test('datasets/KatyPerry.csv', classifier)
    '''
    test('datasets/LMFAO.csv', classifier)
    #main()


   