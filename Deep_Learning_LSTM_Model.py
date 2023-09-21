import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter

# Importing ML Models from Scikit-Learn ML library for Implementation of ML Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

import tensorflow as tf
import keras
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout,SpatialDropout1D,Bidirectional
from keras.callbacks import TensorBoard


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder



from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,precision_score,recall_score,classification_report,confusion_matrix,f1_score
import joblib as jb
import streamlit as st

nltk.download('stopwords', download_dir='nltk_stopwords_data')

nltk.data.path.append('nltk_stopwords_data')

nltk.download('stopwords')

def openFile(path):
    with open(path) as file:
        data = file.read()
    return data
imdb_data = openFile('dataset/imdb_reviews.txt')
amzn_data = openFile('dataset/amazon_reviews.txt')
yelp_data = openFile('dataset/yelp_reviews.txt')


datasets = [imdb_data, amzn_data, yelp_data]

combined_dataset = []
# Separate samples from each other

for dataset in datasets:
    combined_dataset.extend(dataset.split('\n'))

# Separate each label from each sample
dataset = [sample.split('\t') for sample in combined_dataset]

df = pd.DataFrame(data=dataset, columns=['Reviews', 'Labels'])

# Remove any blank reviews
df = df[df["Labels"].notnull()]

df = df.sample(frac=1)

df['Word Count'] = [len(review.split()) for review in df['Reviews']]

df['Uppercase Char Count'] = [sum(char.isupper() for char in review) \
                              for review in df['Reviews']]                           

df['Special Char Count'] = [sum(char in string.punctuation for char in review) \
                            for review in df['Reviews']] 


def getMostCommonWords(reviews, n_most_common, stopwords=None):
    # flatten review column into a list of words, and set each to lowercase
    flattened_reviews = [word for review in reviews for word in \
                         review.lower().split()]

    # remove punctuation from reviews
    flattened_reviews = [''.join(char for char in review if \
                                 char not in string.punctuation) for \
                         review in flattened_reviews]

    # remove stopwords, if applicable
    if stopwords:
        flattened_reviews = [word for word in flattened_reviews if \
                             word not in stopwords]

    # remove any empty strings that were created by this process
    flattened_reviews = [review for review in flattened_reviews if review]

    return Counter(flattened_reviews).most_common(n_most_common)



tokenizer = Tokenizer(num_words=500, split=' ') 
tokenizer.fit_on_texts(df['Reviews'].values)

X = tokenizer.texts_to_sequences(df['Reviews'].values)
X = pad_sequences(X)

# Encoded the target column
lb=LabelEncoder()
df['Reviews'] = lb.fit_transform(df['Reviews'])


print(X.shape[1])


model = Sequential()
model.add(Embedding(5000, 128, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True)))
model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n")
print(model.summary())



#Splitting the data into training and testing
y = pd.get_dummies(df['Labels'], columns=['negative', 'positive'])  # One-hot encode labels

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 42)

tensorboard_callback = TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True, write_images=True)

batch_size=32
model.fit(X_train, y_train, epochs = 5, batch_size=batch_size, verbose = 'auto',callbacks=[tensorboard_callback])

jb.dump(model,'lstm.h5')
jb.dump(tokenizer,'tokenizer.h5')

y_pred=model.predict(X_test)

y_test_1d=np.argmax(y_test.values,axis=1)
y_pred_1d=np.argmax(y_pred,axis=1)

y_test=y_test_1d
y_pred=y_pred_1d
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)



# Calculate precision, recall, and F1-score
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Generate a classification report
class_report = classification_report(y_test, y_pred)

# Print the evaluation metrics
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)



"Predictions"

# Function to predict sentiment class
def predict_sentiment(input_text):
    # Tokenize and pad the input text
    max_len = 48  # This should match the value used during training
    text_sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(text_sequences, maxlen=max_len)

    # Make predictions
    sentiment_classes = ['Negative', 'Positive']
    predictions = model.predict(padded_sequences)
    print(predictions)
    analysis=predictions[0][1]
    if analysis>0.5:
        prediction="Positive"
        return prediction
    elif analysis==0.5:
        prediction="Neutral"
        return prediction
    elif analysis<0.5:
        prediction="Negative"
        return prediction
    return predicted_sentiment

# Get user input
user_input = ""

# Predict sentiment
predicted_sentiment = predict_sentiment(user_input)
print(f"Predicted Sentiment: {predicted_sentiment}")


