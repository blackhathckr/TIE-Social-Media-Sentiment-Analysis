import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from collections import Counter

# Importing Models from Scikit-Learn ML library for Implementation of Algorithms

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.feature_extraction.text import TfidfVectorizer

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

vectorizer = TfidfVectorizer()
bow = vectorizer.fit_transform(df['Reviews'])
labels = df['Labels']

vectorizer = TfidfVectorizer(min_df=15)
bow = vectorizer.fit_transform(df['Reviews'])
X_train, X_test, y_train, y_test = train_test_split(bow, labels, test_size=0.33,random_state=42)




log_reg_hp=logisticRegression_hyperparameters = {
    'penalty': ['l1', 'l2','elasticnet'],        # Regularization type
    'C': [0.001, 0.01, 0.1, 1, 10], # Inverse of regularization strength
    'solver': ['liblinear', 'saga'] # Optimization algorithm
}


decision_tree_hp=decisionTree_hyperparameters = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider when looking for the best split
}

rf_hp=randomForest_hyperparameters = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2']  # Number of features to consider for splitting
}


multinomailNB_hp=multinomialNB_hyperpatameters = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]  # Alpha values for Laplace smoothing
}


bernoulliNB_hp=bernoulliNB_hyperparameters = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]  # Alpha values for Laplace smoothing
}

svc_hp=svc_hyperparameters = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel functions to try
    'degree': [2, 3, 4],  # Degree for the polynomial kernel (if used)
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf' and 'poly'
}

kNeighbors_hp=kNeighbors_hyperparameters = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
    'p': [1, 2],  # Distance metric (1 for Manhattan distance, 2 for Euclidean distance)
}


models=[LogisticRegression,DecisionTreeClassifier,RandomForestClassifier,MultinomialNB,BernoulliNB,SVC,KNeighborsClassifier]

hyperparameters=[log_reg_hp,decision_tree_hp,rf_hp,multinomailNB_hp,bernoulliNB_hp,svc_hp,kNeighbors_hp]

for i in range(7):
      

    # Create a GridSearchCV object with the LogisticRegression model and parameter grid
    grid_search = GridSearchCV(models[i](), hyperparameters[i], cv=5, n_jobs=-1)

    # Fit the grid search to the training data
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters found by the grid search
    best_params = grid_search.best_params_

    # Create a Model with the best hyperparameters
    best_model = models[i](**best_params)

    # Fit the new model to the training data
    best_model.fit(X_train, y_train)


    # Save the model using Joblib

    jb.dump(best_model,"./newmodels/"+str(models[i].__name__)+str("_Model.joblib"))

    # Evaluate the model's performance on the test data
    accuracy = best_model.score(X_test, y_test)

    # Print the best accuracy
    
    print("Accuracy on Test Data:", accuracy)

    y_pred = best_model.predict(X_test)

    # Calculate accuracy score
    accuracy = accuracy_score(y_test, y_pred)
    

    # Calculate other classification metrics
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    class_report = classification_report(y_test, y_pred, target_names=best_model.classes_)
    confusion = confusion_matrix(y_test, y_pred)

    print("\n")
    print("\n")
    print(models[i])
    print("Best Hyperparameters:", best_params) #Best hyperparameters
    print(f'Accuracy Score: {accuracy}')
    print(f'Precision Score: {precision}')
    print(f'Recall Score: {recall}')
    print(f'F1 Score: {f1}')
    print(f'Classification Report:\n{class_report}')
    print(f'Confusion Matrix:\n{confusion}')

    sentence = vectorizer.transform(['Amazing Product'])

    analysis=best_model.predict_proba(sentence)

    print(f'Analysis: {analysis}')
    print(f'Analysis-Negative: {analysis.item(0)}')
    print(f'Analysis-Positive: {analysis.item(1)}')

    analysis=analysis.item(1)
    if analysis>0.5:
            print("Positive")
    elif analysis==0.5:
            print("Neutral")
    else:
            print("Negative")


jb.dump(vectorizer,"vectorizer.joblib")





















# Save the vectorizer 








