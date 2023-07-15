# Standard imports
import numpy as np
import json
import random
import os
# Linguistic imports
import pickle
import nltk
import keras
from nltk.stem import WordNetLemmatizer
# Tensorflow imports
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD, Adam
import keras

lemmatizer = WordNetLemmatizer()

def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'json', 'intents.json')

    with open(file_path) as file:
        intents_data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    def get_child_intents(intent):
        nonlocal words, labels, docs_x, docs_y
        for child_intent in intent.get("childIntents", []):
            for pattern in child_intent['patterns']:
                # tokenize each word in the pattern
                wrds = nltk.word_tokenize(pattern)
                # add to words list
                words.extend(wrds)
                # add to docs_x list
                docs_x.append(wrds)
                # add to docs_y list
                docs_y.append(child_intent['tag'])
            if child_intent['tag'] not in labels:
                labels.append(child_intent['tag'])
            get_child_intents(child_intent)

    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the pattern
            wrds = nltk.word_tokenize(pattern)
            # add to words list
            words.extend(wrds)
            # add to docs_x list
            docs_x.append(wrds)
            # add to docs_y list
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])
        get_child_intents(intent)

    # remove duplicates and lemmatize words list
    words = list(set([lemmatizer.lemmatize(w.lower()) for w in words]))
    # sort labels list
    words.sort()
    
    labels = sorted(labels)
    # create bag of words (bow) for each sentence
    training = []
    out_empty = [0] * len(labels)
    for x, doc in enumerate(docs_x):
        bow = []
        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bow.append(1)
            else:
                bow.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bow + output_row)
    # shuffle training data and convert to numpy array
    training = np.array(training)
    # split training and testing data
    train_x = list(training[:, :-len(labels)])
    train_y = list(training[:, -len(labels):])

    return words, labels, training, train_x, train_y, intents_data

def load_data_no_children():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'json', 'intents.json')

    with open(file_path) as file:
        data = json.load(file)

    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data['intents']:
        for pattern in intent['patterns']:
            # tokenize each word in the pattern
            wrds = nltk.word_tokenize(pattern)
            # add to words list
            words.extend(wrds)
            # add to docs_x list
            docs_x.append(wrds)
            # add to docs_y list
            docs_y.append(intent['tag'])
        if intent['tag'] not in labels:
            labels.append(intent['tag'])

    # remove duplicates and lemmatize words list
    words = [lemmatizer.lemmatize(w.lower()) for w in list(set(words))]
    # sort labels list
    words.sort()
    labels = sorted(labels)
    # create bag of words (bow) for each sentence
    training = []
    out_empty = [0] * len(labels)
    for x, doc in enumerate(docs_x):
        bow = []
        wrds = [lemmatizer.lemmatize(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bow.append(1)
            else:
                bow.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bow + output_row)
    # shuffle training data and convert to numpy array
    training = np.array(training)
    # split training and testing data
    train_x = list(training[:, :-len(labels)])
    train_y = list(training[:, -len(labels):])

    base_intents = []

    for intent in data['intents']:
        new_intent = {key: value for key, value in intent.items() if key != 'childIntents'}
        base_intents.append(new_intent)

    intents_data2 = {'intents': base_intents}

    return words, labels, training, train_x, train_y, intents_data2

def create_model(train_x, train_y):
    
    model = Sequential([
        Dense(180, input_shape=(len(train_x[0]),), activation='relu'),
        Dropout(0.5),
        Dense(80, activation='relu'),
        Dropout(0.5),
        Dense(30, activation='relu'),
        Dense(len(train_y[0]), activation='softmax')
    ])
    
    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    # model = define_model(train_x, train_y)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['categorical_accuracy'])
    model.fit(np.array(train_x), np.array(train_y), epochs=150, batch_size=5, verbose=1)
    return model

def start_model_creation():
    """Create Tensorflow h5 model."""

    # Load data and create models
    words, labels, training, train_x, train_y, intents_data = load_data()
    model = create_model(train_x, train_y)

    words2, labels2, training2, train_x2, train_y2, intents_data2 = load_data_no_children()
    model2 = create_model(train_x2, train_y2)
    
    keras.models.save_model(model, "chatbot_model.h5", overwrite=True)
    keras.models.save_model(model2, "chatbot_model2.h5",overwrite=True)
    

if __name__ == "__main__":
    start_model_creation()
