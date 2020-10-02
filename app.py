from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import random
from textblob import TextBlob, Word
#from sklearn.externals import joblib
#import sklearn.externals.joblib as extjoblib
import joblib
import pickle

# load the model from disk
filename = 'nlp_model.pkl'
clf = joblib.load(open(filename, 'rb'))
cv=joblib.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
	if request.method == 'POST':
		message = request.form['message']
		data = [message]
		vect = cv.transform(data)
        #vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)
        
	blob=TextBlob(text)
        nouns=list()
        for word,tag in blob.tags:
            if tag == 'NN':
                nouns.append(word.lemmatize())
        display=[]
        output=""
        for item in random.sample(nouns,len(nouns)):  
            word=Word(item)
            if word not in display:
                display.append(word.capitalize())
                
        for i in display:
            if len(i) > 2:
                output = output + " " + i
            else:
                output = ""
	return render_template('result.html',prediction = my_prediction,summary = output)



if __name__ == '__main__':
	app.run(debug=True)
