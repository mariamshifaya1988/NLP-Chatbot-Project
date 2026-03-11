
import pickle
import nltk

from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

nltk.download('punkt')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

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

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,2))
X = vectorizer.fit_transform(texts)

model = SVC(kernel='linear', probability=True)
model.fit(X, labels)

# Save model
pickle.dump(model, open("/content/drive/MyDrive/NLP_Chatbot_Project/model.pkl", "wb"))
pickle.dump(vectorizer, open("/content/drive/MyDrive/NLP_Chatbot_Project/vectorizer.pkl", "wb"))

print("Model trained successfully!")

