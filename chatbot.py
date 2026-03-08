
import random
import pickle

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

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
