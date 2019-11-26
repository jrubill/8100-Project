from keras import models, layers, Sequential, optimizers
import csv
import re
import math
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

mode = 'create'
#mode = 'test'
vectorizer = CountVectorizer(max_features=1500)

y = []
x = []
with open('datasets/KatyPerry.csv', "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'COMMENT_ID':
            continue
        content = re.sub(r'[^\w\s]', '', row[3]).lower()
        content = content.split(" ")
        y.append(int(row[4]))
        x.append( ' '.join(content))
with open('datasets/Eminem.csv', "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if row[0] == 'COMMENT_ID':
            continue
        content = re.sub(r'[^\w\s]', '', row[3]).lower()
        content = content.split(" ")
        content = [word for word in content if not isinstance(word, int)]
        y.append(int(row[4]))
        x.append( ' '.join(content))

vectorizer.fit_transform(x).toarray()
vocab_size = len(vectorizer.get_feature_names())
x = vectorizer.fit_transform(x).toarray()
le = LabelEncoder()
y = le.fit_transform(y)
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)

if mode == 'create':
    
    model = Sequential([
        layers.Embedding(vocab_size, 64),
        layers.Conv1D(64,5,activation='relu'),
        layers.MaxPooling1D(5),
        layers.Conv1D(64,5,activation='relu'),
        layers.MaxPooling1D(35),
        layers.Conv1D(64,5,activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='softmax')
    ])
    
    '''
    model = Sequential([
        layers.Embedding(vocab_size, 64),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Bidirectional(layers.LSTM(32)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    '''
    model.compile(loss='binary_crossentropy',
    optimizer=optimizers.Adam(1e-4),
    metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10)
    with open("rnn.pickle", "wb") as f:
        pickle.dump(model, f)
    # print(model.evaluate(x_test, y_test))
else:
    model = 0
    with open("rnn.pickle", "rb") as f:
        model = pickle.load(f)
    wrong = 0
    right = 0
    with open("datasets/LMFAO.csv", "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if row[0] == 'COMMENT_ID':
                continue
            content = re.sub(r'[^\w\s]', '', row[3]).lower()
            #content = content.split(" ")
            if int(row[4]) == int(model.predict(vectorizer.transform([content]))):
                right += 1
            else:
                wrong += 1
        print(f'right: {right}, wrong: {wrong}')
