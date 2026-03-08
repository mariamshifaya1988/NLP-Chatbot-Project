
import pickle
import nltk
import numpy as np

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

texts = []
labels = []

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

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(pattern)
        labels.append(intent["tag"])

# Vectorization
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(texts)

# Train model
model = LogisticRegression()
model.fit(X, labels)

# Save model
pickle.dump(model, open("/content/drive/MyDrive/NLP_Chatbot_Project/model.pkl", "wb"))
pickle.dump(vectorizer, open("/content/drive/MyDrive/NLP_Chatbot_Project/vectorizer.pkl", "wb"))

print("Model trained successfully!")

