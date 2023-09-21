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
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.callbacks import TensorBoard


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



#####################################################################################################################################


# LOGISTIC REGRESSION MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
logisticRegression_hyperparameters = {
    'penalty': ['l1', 'l2','elasticnet'],        # Regularization type
    'C': [0.001, 0.01, 0.1, 1, 10], # Inverse of regularization strength
    'solver': ['liblinear', 'saga'] # Optimization algorithm
}

# Create a GridSearchCV object with the LogisticRegression model and parameter grid
grid_search = GridSearchCV(LogisticRegression(), logisticRegression_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new LogisticRegression model with the best hyperparameters
best_logistic_regression_model = LogisticRegression(**best_params)

# Fit the new model to the training data
best_logistic_regression_model.fit(X_train, y_train)

best_logistic_regression_model.fit(bow,labels)


# Save the Best model using joblib

jb.dump(best_logistic_regression_model,"logisticRegressionModel.joblib")

# Prediction 

y_pred = best_logistic_regression_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_logistic_regression_model.score(X_test, y_test)

# Print the best hyperparameters and accuracy

print("\n")
print("Logistic Regression Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_logistic_regression_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')

with open("models_info.txt","a+") as file:
    file.write("Logistic Regression Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")
    


#####################################################################################################################################


# DECISION TREE MODEL ==========================================================================================================



# Define the hyperparameters and their values to search through
decisionTree_hyperparameters = {
    'criterion': ['gini', 'entropy'],  # Splitting criterion
    'max_depth': [None, 10, 20, 30, 40, 50],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt', 'log2', None]  # Number of features to consider when looking for the best split
}

# Create a GridSearchCV object with the DecisionTreeClassifier model and parameter grid
grid_search = GridSearchCV(DecisionTreeClassifier(), decisionTree_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new DecisionTreeClassifier model with the best hyperparameters
best_decision_tree_model = DecisionTreeClassifier(**best_params)

# Fit the new model to the training data
best_decision_tree_model.fit(X_train, y_train)

best_decision_tree_model.fit(bow,labels)


# Save the Best model using joblib

jb.dump(best_decision_tree_model,"decisionTreeModel.joblib")


# Prediction 

y_pred = best_decision_tree_model.predict(X_test)


# Evaluate the model's performance on the test data
accuracy = best_decision_tree_model.score(X_test, y_test)


# Print the best hyperparameters and accuracy

print("\n")
print("Decision Tree Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_decision_tree_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')


with open("models_info.txt","a+") as file:
    file.write("Decision Tree Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")


#####################################################################################################################################


# RANDOM FOREST MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
""" randomForest_hyperparameters = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4],  # Minimum number of samples required to be at a leaf node
    'max_features': ['auto', 'sqrt']  # Number of features to consider for splitting
} """

randomForest_hyperparameters = {
    'n_estimators': [200],  # Number of trees in the forest
    'max_depth': [None],  # Maximum depth of each tree
    'min_samples_split': [2],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1],  # Minimum number of samples required to be at a leaf node
    'max_features': ['sqrt']  # Number of features to consider for splitting
}

# Create a GridSearchCV object with the RandomForestClassifier model and parameter grid
grid_search = GridSearchCV(RandomForestClassifier(),randomForest_hyperparameters , cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new RandomForestClassifier model with the best hyperparameters
best_random_forest_model = RandomForestClassifier(**best_params)

# Fit the new model to the training data
best_random_forest_model.fit(X_train, y_train)

best_random_forest_model.fit(bow,labels)


# Save the Best model using joblib

jb.dump(best_random_forest_model,"randomForestModel.joblib")

# Prediction 

y_pred = best_random_forest_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_random_forest_model.score(X_test, y_test)


# Print the best hyperparameters and accuracy

print("\n")
print("Random Forest Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_random_forest_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')


with open("models_info.txt","a+") as file:
    file.write("Random Forest Classifier Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")


#####################################################################################################################################



# MULTINOMIAL NB MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
multinomialNB_hyperparameters = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]  # Alpha values for Laplace smoothing
}

# Create a GridSearchCV object with the MultinomialNB model and parameter grid
grid_search = GridSearchCV(MultinomialNB(), multinomialNB_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new MultinomialNB model with the best hyperparameters
best_multinomial_nb_model = MultinomialNB(**best_params)

# Fit the new model to the training data
best_multinomial_nb_model.fit(X_train, y_train)

best_multinomial_nb_model.fit(bow,labels)

# Save the Best model using joblib

jb.dump(best_multinomial_nb_model,"multinomialNB_Model.joblib")

# Prediction 

y_pred = best_multinomial_nb_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_multinomial_nb_model.score(X_test, y_test)


# Print the best hyperparameters and accuracy

print("\n")
print("MultinomialNB Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_multinomial_nb_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')


with open("models_info.txt","a+") as file:
    file.write("Multinomial NB Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")

#####################################################################################################################################



# BERNOULLI NB MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
bernoulliNB_hyperparameters = {
    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0]  # Alpha values for Laplace smoothing
}

# Create a GridSearchCV object with the BernoulliNB model and parameter grid
grid_search = GridSearchCV(BernoulliNB(), bernoulliNB_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new BernoulliNB model with the best hyperparameters
best_bernoulli_nb_model = BernoulliNB(**best_params)

# Fit the new model to the training data
best_bernoulli_nb_model.fit(X_train, y_train)

best_bernoulli_nb_model.fit(bow,labels)

# Save the Best model using joblib

jb.dump(best_bernoulli_nb_model,"bernoulliNB_Model.joblib")


# Prediction 

y_pred = best_bernoulli_nb_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_bernoulli_nb_model.score(X_test, y_test)

# Print the best hyperparameters and accuracy
print("\n")
print("BernoulliNB Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_bernoulli_nb_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')



with open("models_info.txt","a+") as file:
    file.write("Bernoulli NB Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")

#####################################################################################################################################



# SUPPORT VECTOR CLASSIFIER MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
svc_hyperparameters = {
    'C': [0.1, 1, 10],  # Regularization parameter
    'kernel': ['linear', 'rbf', 'poly'],  # Kernel functions to try
    'degree': [2, 3, 4],  # Degree for the polynomial kernel (if used)
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],  # Kernel coefficient for 'rbf' and 'poly'
}

# Create a GridSearchCV object with the SVC model and parameter grid
grid_search = GridSearchCV(SVC(probability=True), svc_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new SVC model with the best hyperparameters
best_svc_model = SVC(**best_params)

# Fit the new model to the training data
best_svc_model.fit(X_train, y_train)

best_svc_model.fit(X_train, y_train)


# Save the Best model using joblib

jb.dump(best_svc_model,"svc_Model.joblib")

# Prediction 

y_pred = best_svc_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_svc_model.score(X_test, y_test)

# Print the best hyperparameters and accuracy
print("\n")
print("Support Vector Classifier (SVC) Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_svc_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')



with open("models_info.txt","a+") as file:
    file.write("Support Vector Classifier ( SVC ) Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")


#####################################################################################################################################


# K NEIGHBORS CLASSIFIER MODEL ==========================================================================================================

# Define the hyperparameters and their values to search through
kNeighborsClassifier_hyperparameters = {
    'n_neighbors': [3, 5, 7, 9],  # Number of neighbors
    'p': [1, 2],  # Distance metric (1 for Manhattan distance, 2 for Euclidean distance)
}

# Create a GridSearchCV object with the KNeighborsClassifier model and parameter grid
grid_search = GridSearchCV(KNeighborsClassifier(), kNeighborsClassifier_hyperparameters, cv=5, n_jobs=-1)

# Fit the grid search to the training data
grid_search.fit(X_train, y_train)

# Get the best hyperparameters found by the grid search
best_params = grid_search.best_params_

# Create a new KNeighborsClassifier model with the best hyperparameters
best_kneighbors_model = KNeighborsClassifier(**best_params)

# Fit the new model to the training data
best_kneighbors_model.fit(X_train, y_train)

best_kneighbors_model.fit(bow,labels)

# Save the Best model using joblib

jb.dump(best_kneighbors_model,"kNeighborsModel.joblib")

# Prediction 

y_pred = best_kneighbors_model.predict(X_test)

# Evaluate the model's performance on the test data
accuracy = best_kneighbors_model.score(X_test, y_test)

# Print the best hyperparameters and accuracy

print("\n")
print("KNeighbors Model")
print("Best Hyperparameters:", best_params)
print("Accuracy on Test Data:", accuracy)


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy Score: {accuracy}')

# Calculate other classification metrics
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
class_report = classification_report(y_test, y_pred, target_names=best_kneighbors_model.classes_)
confusion = confusion_matrix(y_test, y_pred)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1}')
print(f'Classification Report:\n{class_report}')
print(f'Confusion Matrix:\n{confusion}')



with open("models_info.txt","a+") as file:
    file.write("K Neighbors Classifier Model")
    file.write("\nBest Hyperparameters: "+str(best_params))
    file.write("\nAccuracy: "+str(accuracy))
    file.write("\nPrecision Score: "+str(precision))
    file.write("\nRecall Score: "+str(recall))
    file.write("\nF1 Score: "+str(f1))
    file.write("\nClassification Report : "+str(class_report))
    file.write("\nConfusion Matrix: "+str(confusion))
    file.write("\n")
    file.write("\n")
    file.write("\n")
    file.write("\n")

#####################################################################################################################################

# Save the vectorizer 

jb.dump(vectorizer,"vectorizer.joblib")








