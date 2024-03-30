from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle as pk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
import re
from nltk.stem.porter import PorterStemmer
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.models import load_model

app = Flask(__name__)

# Load intents JSON
with open('datasets/intents.json') as json_data:
    intents = json.load(json_data)

# Load model to predict user result
loadedIntentClassifier = load_model('saved_state/intent_model.h5')
loaded_intent_CV = pk.load(open('saved_state/IntentCountVectorizer.sav', 'rb'))

# Load Entity model
loadedEntityCV = pk.load(open('saved_state/EntityCountVectorizer.sav', 'rb'))
loadedEntityClassifier = pk.load(open('saved_state/entity_model.sav', 'rb'))

# Load label maps
with open('saved_state/intent_label_map.json') as label_map_file:
    intent_label_map = json.load(label_map_file)

with open('saved_state/entity_label_map.json') as label_map_file:
    entity_label_map = json.load(label_map_file))

def getEntities(query):
    query = loadedEntityCV.transform(query).toarray()
    response_tags = loadedEntityClassifier.predict(query)
    entity_list=[]
    for tag in response_tags:
        if tag in entity_label_map.values():
            entity_list.append(list(entity_label_map.keys())[list(entity_label_map.values()).index(tag)])
    return entity_list

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_query = request.form['user_query']
    query = re.sub('[^a-zA-Z]', ' ', user_query)
    query = query.split(' ')
    ps = PorterStemmer()
    tokenized_query = [ps.stem(word.lower()) for word in query]
    processed_text = ' '.join(tokenized_query)
    processed_text = loaded_intent_CV.transform([processed_text]).toarray()
    predicted_Intent = loadedIntentClassifier.predict(processed_text)
    result = np.argmax(predicted_Intent, axis=1)
    USER_INTENT = [key for key, value in intent_label_map.items() if value == result[0]][0]
    responses = [i['responses'] for i in intents['intents'] if i['tag'] == USER_INTENT][0]
    bot_response = random.choice(responses)
    entities = getEntities(tokenized_query)
    token_entity_map = dict(zip(entities, tokenized_query))
    return jsonify({'response': bot_response, 'entities': token_entity_map})

if __name__ == '__main__':
    app.run(debug=True)
