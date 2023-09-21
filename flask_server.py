from flask import Flask, render_template,request
import joblib as jb
from keras.preprocessing.sequence import pad_sequences
app=Flask(__name__)

deep_lstm_model=jb.load("./Deep_Learning_LSTM_Model/lstm.h5")

tokenizer=jb.load("./Deep_Learning_LSTM_Model/tokenizer.h5")

vectorizer=jb.load("./ML_Models/vectorizer.joblib")

def predict_sentiment(input_text):
    # Tokenize and pad the input text
    max_len = 48  # This should match the value used during training
    text_sequences = tokenizer.texts_to_sequences([input_text])
    padded_sequences = pad_sequences(text_sequences, maxlen=max_len)

    # Make predictions
    sentiment_classes = ['Negative', 'Positive']
    predictions = deep_lstm_model.predict(padded_sequences)
    #print(predictions)
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

models=["randomForestModel1.joblib",'deep_LSTM.png',"decisionTreeModel.joblib","logisticRegressionModel.joblib","multinomialNB_Model.joblib","bernoulliNB_Model.joblib","svc_Model.joblib","kNeighborsModel.joblib"]

models_info=["randomForest.txt",'deep_LSTM.png',"decisionTree.txt","logisticRegression.txt","multinomialNB.txt","bernoulliNB.txt","svc.txt","kNeighbors.txt"]

model_metrics_img=['randomForest.png','deep_LSTM.png','decisionTree.png','logisticRegression.png','multinomialNB.png','bernoulliNB.png','svc.png','kNeighbors.png']

@app.route("/",methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict",methods=["POST"])
def predict():
    if request.method == "POST":
        model_id=request.form['model']
        model_id=int(model_id)
        text=request.form['text']
        deep_text=request.form['text']
        model_metrics=model_metrics_img[model_id]
        prediction=None
        metrics_height=None 
        metrics_width=None
        if model_id==1:
            model="Deep Learning LSTM ( Long Short Term Memory ) Model"
            with open("./Deep_Learning_LSTM_Model/LSTM_info.txt") as file:
                info=file.readlines()
            analysis=predict_sentiment(deep_text)
            prediction=analysis
            metrics_height="500px"
            metrics_width="1200px"

        elif model_id==6:
            model=models[model_id]
            model_info_file=models_info[model_id]
            model_metrics=model_metrics_img[model_id]
            model=jb.load("./ML_Models/"+str(model))
            with open("./ML_models_info/"+model_info_file) as file:
                info=file.readlines()
            text=vectorizer.transform([text])
            analysis=model.predict(text)
            analysis=int(analysis)
            if analysis==1:
                prediction="Positive"
            elif analysis==0:
                prediction="Negative"""
        else:
            model=models[model_id]
            model_metrics=model_metrics_img[model_id]
            model_info_file=models_info[model_id]
            model=jb.load("./ML_Models/"+str(model))
            with open("./ML_models_info/"+model_info_file) as file:
                info=file.readlines()
            text=vectorizer.transform([text])
            analysis=model.predict_proba(text)
            print("Analysis :",analysis)
            print("Positive :",analysis.item(1))
            print("Negative :",analysis.item(0))
            analysis=analysis.item(1)
            if analysis>0.5:
                prediction="Positive"
            elif analysis==0.5:
                prediction="Neutral"
            elif analysis<0.5:
                prediction="Negative"

        return render_template("prediction.html",model=model,prediction=prediction,model_metrics=model_metrics,metrics_height=metrics_height,metrics_width=metrics_width)

if __name__ == '__main__':
    app.run(debug=True)