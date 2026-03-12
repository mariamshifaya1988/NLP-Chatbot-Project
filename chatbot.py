

import pickle
import random
import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

intents = {
 "intents":[
   {
     "tag":"greeting",
     "responses":["Hello!","Hi there!","Hey!"]
   },
   {
     "tag":"goodbye",
     "responses":["Goodbye!","See you later"]
   },
   {
     "tag":"thanks",
     "responses":["You're welcome","Happy to help","Anytime!"]
   }
 ]
}

def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

def get_response(user_input):

    processed = preprocess(user_input)

    vec = vectorizer.transform([processed])

    predicted_tag = model.predict(vec)[0]

    for intent in intents["intents"]:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])

    return "Sorry, I didn't understand."
