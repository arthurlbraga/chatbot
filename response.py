# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
from keras.models import load_model
import tensorflow as tf
import random

import pickle
import json
import pandas as pd

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)

    return(np.array(bag))

def classify(sentence):
    inp = pd.DataFrame([bow(sentence, words)], dtype=float, index=['input'])
    # generate probabilities from the model
    results = model.predict([inp])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
    # return tuple of intent and probability
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    print(results)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details: print ('context:', i['context_set'])
                        context[userID] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                        (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                        if show_details: print ('tag:', i['tag'])
                        # a random response from the intent
                        return random.choice(i['responses'])
                    else:
                        return random.choice(intents['intents'][0]['responses'])

            results.pop(0)

# restore all of our data structures
data = pickle.load( open( "training_data", "rb" ) )
words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
with open('intents.json') as json_data:
    intents = json.load(json_data)

# load model
model = load_model('shop_model.h5p')

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25

print('Start conversation \n\n')
while True:
    inp = input()

    if(inp == 'quit'):
        break

    print('Me: ' + inp)
    reply = response(inp)
    print(reply)
    print('Bot: ' + reply)


# print(classify('is your shop open today?'))
# print(response('is your shop open today?'))
# print(response('we want to rent a moped'))
# print(response('today'))
# print(response("Hi there!", show_details=True))
# print(response('today'))
# print(classify('today'))
# print(response("thanks, your great"))
