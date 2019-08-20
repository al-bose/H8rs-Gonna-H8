import csv
import numpy as np
import pandas as pd
import re
import sklearn
import string
import nltk
from nltk import stopwords
from sklearn.externals import joblib
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from string import punctuation



#function that is passed as a callable to the CountVectorizer analyzer parameter
#generates a list of all unigrams while removing stop words as defined in the 
def generate_unigrams(text):

	list_of_c_punctuation = string.punctuation
	stop_words = stopwords.words('english')
	unigrams = []

	punctuation_removed = [char for char in list(text) if char not in list_of_c_punctuation]
	punctuation_removed = ''.join(punctuation_removed)

	return [word for word in punctuation_removed.split() if word.lower() not in stop_words]

#function used to clean each text field in the tweet dataframe
def sanitize(text):
	# Remove HTML special entities (e.g. &amp;)
	text = re.sub(r'\&\w*;', '', text)
	#Convert @username to AT_USER
	text = re.sub('@[^\s]+','',text)
	# Remove tickers
	text = re.sub(r'\$\w*', '', text)
	# To lowercase
	text = text.lower()
	# Remove hyperlinks
	text = re.sub(r'https?:\/\/.*\/\w*', '', text)
	# Remove hashtags
	text = re.sub(r'#\w*', '', text)
	# Remove Punctuation and split 's, 't, 've with a space for filter
	text = re.sub(r'[' + string.punctuation.replace('@', '') + ']+', ' ', text)
	# Remove words with 2 or fewer letters
	text = re.sub(r'\b\w{1,2}\b', '', text)
	# Remove whitespace (including new line characters)
	text = re.sub(r'\s\s+', ' ', text)
	# Remove single space remaining at the front of the text.
	text = text.lstrip(' ') 
	# Remove characters beyond Basic Multilingual Plane (BMP) of Unicode:
	text = ''.join(c for c in text if c <= '\uFFFF') 

	return text


def main():

	#load the generated model and the test data 
	sentiment_model = joblib.load("sentiment_analysis_model.pkl")
	test_data_raw = pd.read_csv("test_data.csv", encoding="ISO-8859-1", names = ["label", "ids", "date", "flag", "user", "text"])
	test_data_raw = test_data_raw.drop(["ids", "date", "flag", "user",], axis = 1)

	#clean all the text using the sanitize function
	test_data_raw['text'] = test_data_raw['text'].apply(sanitize)

	#remove all labels that are marked neutral because our training dataframe does not have any neutral examples (label = 2)
	test_data_raw = test_data_raw[test_data_raw.label != 2]

	#assign x and y for test data
	x_test = test_data_raw['text']
	y_test = test_data_raw['label']

	#generate predictions 
	y_pred = sentiment_model.predict(x_test)
	target_names = ['Negative','Positive']

	#use sklearn.metrics to report the accuracy score, the confusion matrix and the classification report with class labels
	print('accuracy score: ',accuracy_score(y_test, y_pred))
	print('\n')
	print('confusion matrix: \n',confusion_matrix(y_test,y_pred))
	print('\n')
	print(classification_report(y_test, y_pred, target_names=target_names))



if __name__ == "__main__":
	main()