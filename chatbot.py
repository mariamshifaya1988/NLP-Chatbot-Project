
import json
import pickle
import random

model = pickle.load(open("/content/drive/MyDrive/NLP_Chatbot_Project/model.pkl","rb"))
vectorizer = pickle.load(open("/content/drive/MyDrive/NLP_Chatbot_Project/vectorizer.pkl","rb"))

with open("/content/drive/MyDrive/NLP_Chatbot_Project/intents.json") as file:
    data = json.load(file)

def get_response(user_input):

    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]

    for i in data["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
