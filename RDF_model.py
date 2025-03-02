import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk 
import scipy.sparse as sp
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier # Random Forest Classifier
from sklearn.metrics import accuracy_score, classification_report
# download the stopwords
nltk.download('stopwords') 
stop_words = set(stopwords.words('english'))

# Load the dataset
df = pd.read_csv("IMDB Dataset.csv")

# Preprocess the data
# change the ojbect to lower case and remove the sepcial characters
import re
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text
df['review']=df['review'].apply(preprocess_text)

# Function to remove stopwords
def remove_stopwords(text):
    words = re.findall(r'\w+', text.lower())  # Tokenize text
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)
df['review'] = df['review'].apply(remove_stopwords) # Apply stopword removal

# change the sentiment to 1 and 0
df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})


# Apply TF-IDF vectorization with a limited feature size to reduce memory usage
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['review'])

x=X
y=df['sentiment']

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)

# Predict the sentiment of the test set
y_pred = clf.predict(x_test)

# Calculate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy * 100))

# Classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Save the trained model
import pickle   
with open("random_forest_sentiment.pkl", "wb") as model_file:
    pickle.dump(clf, model_file)

print("Model saved to random_forest_sentiment.pkl successfully!")

# Save the trained vectorizer
with open("tfidf_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("Vectorizer saved to tfidf_vectorizer.pkl successfully!")
