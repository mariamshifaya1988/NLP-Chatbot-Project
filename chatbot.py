
import pickle
import random

model = pickle.load(open("/content/drive/MyDrive/NLP_Chatbot_Project/model.pkl","rb"))
vectorizer = pickle.load(open("/content/drive/MyDrive/NLP_Chatbot_Project/vectorizer.pkl","rb"))

intents = {
 "intents":[
   {
     "tag":"greeting",
     "patterns":["Hi","Hello","Hey"],
     "responses":["Hello!","Hi there!"]
   },
   {
     "tag":"goodbye",
     "patterns":["Bye","See you"],
     "responses":["Goodbye!","See you later"]
   },
   {
     "tag":"thanks",
     "patterns":["Thanks","Thank you"],
     "responses":["You're welcome"]
   }
 ]
}
def get_response(user_input):

    vec = vectorizer.transform([user_input])
    intent = model.predict(vec)[0]

    for intent in intents["intents"]:
        if i["tag"] == intent:
            return random.choice(i["responses"])
