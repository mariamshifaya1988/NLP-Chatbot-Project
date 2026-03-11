
import pickle
import random

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
    predicted_intent = model.predict(vec)[0]

    for intent_data in intents["intents"]:
        if intent_data["tag"] == predicted_intent:
            return random.choice(intent_data["responses"])
    return "Sorry, I didn't understand."     
