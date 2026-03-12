

import pickle
import json
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# intents dataset
intents = {
 "intents":[
   {
     "tag":"greeting",
     "patterns":["Hi","Hello","Hey","Hi there","Hello there","Good morning","Good evening","Hey buddy"],
     "responses":["Hello!","Hi there!","Hey!"]
   },
   {
     "tag":"goodbye",
     "patterns":["Bye","Goodbye","See you","See you later","Catch you later","Talk to you later"],
     "responses":["Goodbye!","See you later"]
   },
   {
     "tag":"thanks",
     "patterns":["Thanks","Thank you","Thanks a lot","Thank you very much","I appreciate it","Much appreciated","Thanks buddy","Thanks for the help","Thank you so much"],
     "responses":["You're welcome","Happy to help","Anytime!"]
   }
 ]
}

def preprocess(text):
    words = nltk.word_tokenize(text.lower())
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)

texts = []
labels = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        texts.append(preprocess(pattern))
        labels.append(intent["tag"])

# vectorizer
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(texts)

# classifier
model = SVC(kernel="linear", probability=True)
model.fit(X, labels)

# save files
pickle.dump(model, open("/content/drive/MyDrive/NLP_Chatbot_Project/model.pkl","wb"))
pickle.dump(vectorizer, open("/content/drive/MyDrive/NLP_Chatbot_Project/vectorizer.pkl","wb"))

print("Model trained successfully")

