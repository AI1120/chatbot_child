import os
import string
import joblib
import spacy
import numpy as np
import tensorflow as tf
import json
import pickle
# import spacy
import random
import time
import re
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
import keras
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout
from word2number import w2n

import datetime
from collections import OrderedDict
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from collections import defaultdict
from keras.backend import manual_variable_initialization 
from create_model import create_model, load_data, load_data_no_children

nlp = spacy.load("en_core_web_lg")

lemmatizer = WordNetLemmatizer()

def bag_of_words(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [0] * len(words)
    for word in sentence_words:
        for i, w in enumerate(words):
            if w == word:
                bag[i] = 1
    return np.array(bag)

def bag_of_words_children(message, words, child_patterns):
    message_words = nltk.word_tokenize(message)
    bag = np.zeros(len(words))
    for idx, w in enumerate(words):
        if w in message_words:
            bag[idx] = 1
    child_bag = np.zeros(len(child_patterns))
    for idx, w in enumerate(child_patterns):
        if w in message_words:
            child_bag[idx] = 1
    return np.concatenate((bag, child_bag))

def search_intents(intents, last_intent):
    for intent in intents:
        if intent["tag"] == last_intent:
            return intent
        if "childIntents" in intent:
            found_intent = search_intents(intent["childIntents"], last_intent)
            if found_intent is not None:
                return found_intent
    return None

def find_most_likely_intent(message, model2, words2, labels2, intents_data2, user_info=None, sol_info=None,
                            last_intent=None):
    clean_message = message.lower()
    clean_message = re.sub(r'[^\w\s]', '', clean_message)
    clean_message = clean_message.replace("idk", "i dont know")

    results = model2.predict(np.array([bag_of_words(clean_message, words2)]))[0]
    most_likely_intent = labels2[np.argmax(results)]
    probability = results[np.argmax(results)]
    print(f"RegIntent-check1: Most likely intent: {most_likely_intent}, Probability: {probability}")

    hello_check = False
    if last_intent in ["hello", "greeting"]:
        hello_check = True

    if probability >= 0.5:
        for intent in intents_data2['intents']:
            if intent['tag'] == most_likely_intent:
                print(f"Found intent: {intent['tag']}")
                last_intent = most_likely_intent
                function_data = intent.get('function', None)
                function_response = None
                updated_last_intent = None
                if function_data:
                    function_name = function_data.split('.')[-1]
                    if function_name in globals():
                        function_response, last_intent = globals()[function_name](clean_message, user_info)
                responses = intent['responses']
                if function_response and responses:
                    response = f"{random.choice(responses)} {function_response}"
                elif function_response and not responses:
                    response = function_response
                elif responses:
                    response = random.choice(responses)
                else:
                    response = "child: Sorry, I don't know how to respond to that."
                response = replace_placeholders(response, user_info, sol_info)
                if hello_check and last_intent == "hello":
                    response, last_intent = conversation_starter()
                return response, last_intent

    fallback_intent = next((intent for intent in intents_data2['intents'] if intent['tag'] == 'fallback'), None)
    if fallback_intent:
        response = random.choice(fallback_intent['responses'])
        response = replace_placeholders(response, user_info, sol_info)
        return response, last_intent

last_intent = ""

def get_child_response(clean_message, message, model, words, labels, intents_data, model2, words2, labels2,
                       intents_data2, user_info=None,
                       sol_info=None, last_intent=None):
    clean_message = re.sub(r'[^\w\s]', '', clean_message)
    clean_message = clean_message.replace("idk", "i dont know")
    # response = None

    response = None

    if last_intent:
        print(f"This is the last intent {last_intent}")
        matched_intent = search_intents(intents_data["intents"], last_intent)

        child_patterns = []
        child_intent_check = matched_intent.get('childIntents', [])

        if child_intent_check:
            child_intent_tags = [child['tag'] for child in matched_intent.get('childIntents', [])]
            if child_intent_tags:
                filtered_labels = [label for label in child_intent_tags if label in labels]
                results = model.predict(np.array([bag_of_words_children(clean_message, words, child_patterns)]))[0]
            else:
                filtered_labels = labels
                results = model.predict(np.array([bag_of_words(clean_message, words)]))[0]

            filtered_results = [results[labels.index(label)] for label in filtered_labels]
            most_likely_intent = filtered_labels[np.argmax(filtered_results)]
            if clean_message in ["yes", "no", "good", "yeah", "yep", "okay", "nope"]:
                probability = filtered_results[np.argmax(filtered_results)] * 6
                probability = 0.50
            else:
                probability = filtered_results[np.argmax(filtered_results)] * 5

            print(f"Child-check: Most likely intent: {most_likely_intent}, Probability: {probability}")

            if probability >= 0.30:
                for intent in matched_intent['childIntents']:
                    if intent['tag'] == most_likely_intent:
                        last_intent = most_likely_intent
                        print(last_intent)
                        function_data = intent.get('function', None)
                        function_response = None
                        updated_last_intent = None

                        if function_data:
                            function_name = function_data.split('.')[-1]
                            f_append = None

                            if '-' in function_name:
                                split_function = [i for i in function_name.split('-') if i]
                                function_name = split_function[0]

                                if len(split_function) > 1:
                                    f_append = split_function[1]

                            try:
                                if function_name in globals():
                                    if f_append:
                                        function_response, updated_last_intent = globals()[function_name](message,
                                                                                                          clean_message,
                                                                                                          model, words,
                                                                                                          labels,
                                                                                                          intents_data,
                                                                                                          model2,
                                                                                                          words2,
                                                                                                          labels2,
                                                                                                          intents_data2,
                                                                                                          last_intent,
                                                                                                          f_append)
                                    else:
                                        response, last_intent = globals()[function_name](clean_message, user_info)
                                    function_response = response
                            except Exception as e:
                                function_response = None
                                updated_last_intent = None

                        responses = intent['responses']
                        if function_response and responses:
                            function_response = function_response[0].lower() + function_response[1:]
                            response = f"{random.choice(responses)}, {function_response}"
                            response = response.replace(".,", ",").replace("!,", ",").replace("..", ".")
                        elif function_response and not responses:
                            response = function_response
                        elif responses:
                            response = random.choice(responses)
                        else:
                            response = "child: Sorry, I don't know how to respond to that."

                        response = replace_placeholders(response, user_info, sol_info)
                        return response, last_intent
            else:
                print("Normal response 2")
                response, last_intent = find_most_likely_intent(clean_message, model2, words2, labels2, intents_data2,
                                                                user_info,
                                                                sol_info,
                                                                last_intent)
                return response, last_intent
        else:
            print("Normal response 3")
            response, last_intent = find_most_likely_intent(clean_message, model2, words2, labels2, intents_data2,
                                                            user_info,
                                                            sol_info,
                                                            last_intent)
            return response, last_intent
    else:
        print("Normal response 4")
        response, last_intent = find_most_likely_intent(clean_message, model2, words2, labels2, intents_data2,
                                                        user_info,
                                                        sol_info,
                                                        last_intent)
        return response, last_intent

def get_random_response(tag, intents_data=None, user_info=None, sol_info=None):
    intent = search_intents(intents_data['intents'], tag)
    if intent is None:
        return None, None
    responses = intent['responses']
    function_data = intent.get('function', None)
    if function_data:
        function_name = function_data.split('.')[-1]
        if function_name in globals():
            globals()[function_name](user_info)
            return None, None
    if responses:
        response = random.choice(responses)
    else:
        response = "child: Sorry, I don't know how to respond to that."
    response = replace_placeholders(response, user_info, sol_info)
    last_intent = intent['tag']
    return response, last_intent

def replace_placeholders(response, user_info, sol_info):
    if user_info is None and sol_info is None:
        return response

    matches = re.findall(r'\{.*?\}', response)

    for match in matches:
        key = match.strip('{}').strip()
        if key.endswith('_all'):
            key = key.replace('_all', '')
            values = user_info.get(key, []) + sol_info.get(key, [])
            if values:
                value = ', '.join(values)
            else:
                value = ''
        elif "sol_mood" in key:
            key = key.replace('sol_mood', '"this is sol_mood for now"')
            values = key
            value = values
        elif "_rnd" in key:
            n = int(key.split('_rnd')[-1])
            key = key.replace(f'_rnd{n}', '')
            values = user_info.get(key, []) + sol_info.get(key, [])
            if values:
                chosen_values = random.sample(values, min(n, len(values)))
                value = ', '.join(chosen_values)
            else:
                value = ''
        else:
            value = user_info.get(key, sol_info.get(key, ''))
        response = response.replace(match, str(value))

    return response

def conversation_starter(clean_message=None, user_info=None):
    words, labels, training, train_x, train_y, intents_data = load_data()

    with open('json/user_info.json', 'r', encoding='utf-16') as f:
        user_info = json.load(f)['user_info']

    with open('json/sol_info.json') as f:
        sol_info = json.load(f)['sol_info']

    with open('json/user_daily_info.json') as f:
        user_daily_info = json.load(f)['daily_info']

    today_goals_starter = get_random_response('sp_today_goals', intents_data, user_info=user_info, sol_info=sol_info)[0]
    focusing_starter = get_random_response('sp_focusing', intents_data, user_info=user_info, sol_info=sol_info)[0]
    what_to_talk_about_starter = \
        get_random_response('what_to_talk_about', intents_data, user_info=user_info, sol_info=sol_info)[0]

    response_intent_mapping = {
        today_goals_starter: 'sp_today_goals',
        focusing_starter: 'sp_focusing',
        what_to_talk_about_starter: 'what_to_talk_about'
    }

    starter_dict = {
        id(today_goals_starter): 'today_goals_starter',
        id(focusing_starter): 'focusing_starter',
        id(what_to_talk_about_starter): 'what_to_talk_about'
    }

    starter_list = [
        today_goals_starter,
        focusing_starter,
        what_to_talk_about_starter
    ]

    random_starter = random.choice(starter_list)
    if random_starter == what_to_talk_about_starter:
        global topic_mode
        topic_mode = True

    random_starter_name = starter_dict[id(random_starter)]

    response = random_starter
    last_intent = response_intent_mapping[response]

    return response, last_intent

def startup():
    chatV2()
    
from keras.optimizers import SGD  

def chatV2(from_saved_model=True):
    manual_variable_initialization(True)
    if from_saved_model:

        # Load Intent file
        words, labels, training, train_x, train_y, intents_data = load_data()
        
        # Create No Children Data 2
        words2, labels2, training2, train_x2, train_y2, intents_data2 = load_data_no_children()        
        
        # load models
        model = tf.keras.models.load_model("chatbot_model.h5")
        # model.summary()
        model2 = tf.keras.models.load_model("chatbot_model2.h5")
        # model2.summary()
       
    else:
        
        # Create models
        words, labels, training, train_x, train_y, intents_data = load_data()
        model = create_model(train_x, train_y)
        # model.summary()
        
        words2, labels2, training2, train_x2, train_y2, intents_data2 = load_data_no_children()
        model2 = create_model(train_x2, train_y2)
        # model2.summary()
                
        tf.keras.models.save_model(model, "chatbot_model.h5", overwrite=True)
        tf.keras.models.save_model(model2, "chatbot_model2.h5",overwrite=True)
        
    
    with open('json/user_info.json', 'r', encoding='utf-16') as f:
        user_info = json.load(f)['user_info']
    # print('user_info : ', user_info)
    with open('json/sol_info.json') as f:
        sol_info = json.load(f)['sol_info']
    # print("sol_info : ", sol_info)
    with open('json/user_daily_info.json') as f:
        user_daily_info = json.load(f)['daily_info']
    # print("user_daily_info", user_daily_info)
    last_intent = None

    # Normal chat loop
    while True:
        # Get user message
        message = input("You: ")

        clean_message = message
        # print('words::', words)
        # print('words2::', words2)
        response, last_intent = get_child_response(clean_message, message, model, words, labels, intents_data, model2,
                                                   words2, labels2, intents_data2, user_info=user_info,
                                                   sol_info=sol_info,
                                                   last_intent=last_intent)
        #return response, last_intent

        # Print bot's response
        print(f"Bot: {response}")



startup()
