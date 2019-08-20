import nltk
import pandas as pd
import re
import sklearn
import string
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


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

	#load the csv file into a pandas dataframe and drop irrelevant columns
	train_data_raw = pd.read_csv("train_data.csv", encoding="ISO-8859-1", names = ["label", "ids", "date", "flag", "user", "text"])
	train_data_raw = train_data_raw.drop(["ids", "date", "flag", "user",], axis = 1)

	#Initial Data Exploration:
	#print(train_data_raw.loc[train_data_raw['label'] == 4].head(5))
	#positives = train_data_raw.loc[train_data_raw['label'] == 4]
	#negatives = train_data_raw.loc[train_data_raw['label'] == 0]
	#Noticed that there are a total of 1.6 million tweets half of which are labeled negative (0) and half of which are labeled positive (4)

	#Testing the sanitize and generate_unigrams functions
	#train_data_raw_sample = train_data_raw[:5]
	#train_data_raw_sample['text']= train_data_raw_sample['text'].apply(sanitize)
	#train_data_raw_sample['unigrams'] = train_data_raw_sample['text'].apply(generate_unigrams)
	#print(train_data_raw_sample.head(5))


	#Test use of CountVectorizer to generate a BOW model for all tweets which will be used later to generate tf-idf scores
	#vectorizer = CountVectorizer(analyzer=generate_unigrams)
	#bow_transformer = vectorizer.fit(train_data_raw['text'])
	#original_text = train_data_raw['text']
	#vectorized_text = bow_transformer.transform(train_data_raw['text'])

	#Test use of TfidfTransformer to calculate the tf-idf scores that will be used later to train the model
	#tfidf = TfidfTransformer()
	#tfidf_transformer = tfidf.fit(vectorized_text)
	#tfidf_scores = tfidf_transformer.transform(vectorized_text)


	#pre process all tweets
	train_data_raw['text'] = train_data_raw['text'].apply(sanitize)


	#Using the full dataframe this model takes a very long time to run on a regular processor
	#Can use the following to run on only a small amount of the training data by changing the number of rows looked at (5000 in this example)
	#X_train, X_test, y_train, y_test = train_test_split(train_data_raw['text'][:5000], train_data_raw['label'][:5000])


	X_train = train_data_raw['text']
	y_train = train_data_raw['label']



	# Using a pipeline makes the code clearer to understand and allows us to do all the pre-processing in 1 step
	#list of name and transform tuples
	#also allows us to only apply the transforms to the training fold during cross validation avoiding overfitting
	pipeline = Pipeline([
		('bow', CountVectorizer(analyzer=generate_unigrams,strip_accents='ascii',
							stop_words='english',
							lowercase=True)),
		('tfidf', TfidfTransformer()), 
		('classifier', MultinomialNB()),
	])


	#this is where we define the values for GridSearchCV to iterate over
	#for bow we choose either unigrams or both unigrams and bigrams
	#tfidf is either true or false
	#classifier alpha parameter is also tuned for smoothing zero probabilities
	parameters = {'bow__ngram_range': [(1, 1), (1, 2)],
			  'tfidf__use_idf': (True, False),
			  'classifier__alpha': (1e-2, 1e-3),
			 }


	# 10-fold cross validation is used to train the model with GridSearch
	grid = GridSearchCV(pipeline, cv=10, param_grid=parameters, verbose=1)
	grid.fit(X_train,y_train)

	# summary of the results of all parameter combinations which is 8 combinations in this case
	print("\nBest Model: %f using %s" % (grid.best_score_, grid.best_params_))
	print('\n')

	means = grid.cv_results_['mean_test_score']
	stds = grid.cv_results_['std_test_score']
	params = grid.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("Mean: %f Stdev:(%f) with: %r" % (mean, stdev, param))


	#save the model for future use
	joblib.dump(grid, "sentiment_analysis_model.pkl")


if (__name__ == "__main__"):
	main()

