import nltk
import json
import pickle
import numpy as np
import random

from nltk.stem import WordNetLemmatizer
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD


tokenized_words = []
classes = []
documents = []
ignore_symbols = ['?', '!', ',', '-', ':']
intents_json = open('intents.json', 'r').read()
intents = json.loads(intents_json)

for intent in intents['intents']:
    for pattern in intent['patterns']:

        words = nltk.word_tokenize(pattern)
        tokenized_words.extend(words)

        documents.append((words, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])


lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(word.lower()) for word in tokenized_words if word not in ignore_symbols]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))


training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words if word not in ignore_symbols]
    for word in words:
        bag.append(1) if word in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype='object')
train_x = list(training[:, 0])
train_y = list(training[:, 1])


model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

sgd = SGD(learning_rate=0.01, weight_decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('model.h5')
