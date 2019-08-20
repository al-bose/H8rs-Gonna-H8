import chart_studio
import chart_studio.plotly as py
import csv
import json
import nltk
import numpy as np
import pandas as pd
import plotly
import plotly.graph_objs as go 
import re
import sklearn
import string
import sys
import tweepy
from IPython.display import IFrame
from nltk.corpus import stopwords
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
from sklearn.externals import joblib
from string import punctuation



#function that is passed as a callable to the CountVectorizer analyzer parameter
#generates a list of all unigrams while removing stop words as defined in the nltk.corpus
def generate_unigrams(text):

	list_of_c_punctuation = string.punctuation
	stop_words = stopwords.words('english')
	unigrams = []

	punctuation_removed = [char for char in list(text) if char not in list_of_c_punctuation]
	punctuation_removed = ''.join(punctuation_removed)

	return [word for word in punctuation_removed.split() if word.lower() not in stop_words]

#function used to change predicted labels to -1 and 1 to help calculate the sentiment score
def switch_labels(label):
	if label == 0:
		return -1
	else:
		return 1


#function that is given the tweet.user.location and takes a best effort approach to identify the state the location belongs to or returns NULL when unable to do so
def get_state_name(location):

	states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
			  "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
			  "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
			  "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
			  "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

	states_dict = {
			'AK': 'ALASKA','AL': 'ALABAMA','AR': 'ARKANSAS',
			'AZ': 'ARIZONA','CA': 'CALIFORNIA','CO': 'COLORADO','CT': 'CONNECTICUT',
			'DC': 'DISTRICT OF COLUMBIA','DE': 'DELAWARE','FL': 'FLORIDA','GA': 'GEORGIA'
			,'HI': 'HAWAII','IA': 'IOWA','ID': 'IDAHO','IL': 'ILLINOIS',
			'IN': 'INDIANA','KS': 'KANSAS','KY': 'KENTUCKY','LA': 'LOUISIANA',
			'MA': 'MASSACHUSETTS','MD': 'MARYLAND','ME': 'MAINE','MI': 'MICHIGAN',
			'MN': 'MINNESOTA','MO': 'MISSOURI',
			'MS': 'MISSISSIPPI','MT': 'MONTANA','NC': 'NORTH CAROLINA',
			'ND': 'NORTH DAKOTA','NE': 'NEBRASKA','NH': 'NEW HAMPSHIRE','NJ': 'NEW JERSEY',
			'NM': 'NEW MEXICO','NV': 'NEVADA','NY': 'NEW YORK','OH': 'OHIO','OK': 'OKLAHOMA',
			'OR': 'OREGON','PA': 'PENNSYLVANIA','RI': 'RHODE ISLAND',
			'SC': 'SOUTH CAROLINA','SD': 'SOUTH DAKOTA','TN': 'TENNESSEE','TX': 'TEXAS',
			'UT': 'UTAH','VA': 'VIRGINIA','VT': 'VERMONT',
			'WA': 'WASHINGTON','WI': 'WISCONSIN','WV': 'WEST VIRGINIA','WY': 'WYOMING'
	}


	location = str(location)
	location = location.replace(',', ' ')
	location = location.split()

	location = [loc.upper() for loc in location]

	for i in range(len(location)):
		try:
			if location[i] in states:
				return location[i]
			elif location[i] in states_dict.values():
				return list(states_dict.keys())[list(states_dict.values()).index(location[i])]
			elif (location[i] == "NEW") and (location[i+1] == "HAMPSHIRE"):
				return "NH"
			elif (location[i] == "NEW") and (location[i+1] == "JERSEY"):
				return "NJ"
			elif (location[i] == "NEW") and (location[i+1] == "MEXICO"):
				return "NM"
			elif (location[i] == "NORTH") and (location[i+1] == "CAROLINA"):
				return "NC"
			elif (location[i] == "NORTH") and (location[i+1] == "DAKOTA"):
				return "ND"
			elif (location[i] == "SOUTH") and (location[i+1] == "CAROLINA"):
				return "SC"
			elif (location[i] == "SOUTH") and (location[i+1] == "DAKOTA"):
				return "SD"
			elif (location[i] == "DISTRICT") and (location[i+1] == "OF") and (location[i+2] == "COLUMBIA"):
				return "DC"
			elif (location[i] == "NEW") and (location[i+1] == "YORK"):
				return "NY"
			elif (location[i] == "RHODE") and (location[i+1] == "ISLAND"):
				return "RI"
			elif (location[i] == "WEST") and (location[i+1] == "VIRGINIA"):
				return "WV"
		except Exception:
			return "NULL"

	return "NULL"



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

#remove the keywords from the tweet to prevent bias towards the topics
def remove_keywords(text,keyword):

	keyword_list = keyword.split("AND")

	#Noticed that my training and test datasets only use ' instead of ’
	#The ’ character is also not in string.punctuation so I manually add it and remove all instances
	list_of_c_punctuation = string.punctuation
	list_of_c_punctuation = list_of_c_punctuation + '’'
	text = [char for char in list(text) if char not in list_of_c_punctuation]
	text = ''.join(text)

	text_list = text.split()
	text_list = [word for word in text_list if word not in keyword_list]

	return(' '.join([str(item) for item in text_list]))



#function that uses the user inputted keyword to query the Twitter API 
def get_tweets(keyword):

	#OAuth 1a authentication
	#PLEASE FILL IN THE FOLLOWING FIELDS WITH YOUR PERSONAL TOKENS!
	consumer_token = "vrvfXhXLzU6UNsKwKNVagiztD"
	consumer_secret = "TkmLEb9MnMSy9wi6bf7v2T4scxVsLF5jtYhdw5ao5mJuOaXHZQ"
	access_token = "2212483580-gBH5DBs6Ki9YfjZpOeGBmzZd1NNC5GC8TEhla8v"
	access_token_secret = "VIx6Mx2wB6xZsf3CiiCw1uxR2QRhTdgVWF83gNpsUhpXg"

	auth = tweepy.OAuthHandler(consumer_token, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

	#create a csv file and use a writer to append the data we will query
	csvFile = open(keyword + '_raw_data.csv', 'w', encoding='utf-8')
	csvWriter = csv.writer(csvFile)

	#query Twitter API using tweepy
	#please feel free to change the parameter passed into items to fetch more tweets if desired
	tweet_num = 0
	for tweet in tweepy.Cursor(api.search,q=keyword,count=1000000,lang="en",tweet_mode="extended").items(10000):
		try:
			# write data to csv
			csvWriter.writerow([tweet.user.location,tweet.full_text])
			tweet_num += 1
			
		except Exception:
			pass

	print(str(tweet_num) + " relevant tweets retrieved!")

	


def main():

	#generate keyword from command line arguments in proper format
	keyword = ''
	for i in range(1,len(sys.argv)):
		keyword = keyword + sys.argv[i].lower() + " AND "

	keyword = keyword.rstrip( " AND ")

	#request relevant tweets and store into a csv file for potential future use
	#If csv file already exists then feel free to comment out these 2 lines
	print("Fetching Tweets...")
	get_tweets(keyword)

	#read the csv file
	col_names=['location','message']
	data = pd.read_csv(keyword + '_raw_data.csv', names=col_names)


	#pre-process the messages to remove unnecessary characters and also the key word(s)
	data['message'] = data['message'].apply(sanitize)
	data['message'] = data['message'].apply(lambda r: remove_keywords(r,keyword))

	#load the analysis model
	sentiment_model = joblib.load("sentiment_analysis_model.pkl")

	#try to identify the state the tweet belongs to and remove all unidentified tweets
	data['state'] = data['location'].apply(get_state_name)
	data = data[data.state != "NULL"]

	valid_tweets = len(data.index)
	print(str(valid_tweets) + " valid tweets found!")


	#use model to generate the sentiment predictions
	print ("Analyzing Tweets ...")
	data['prediction'] = sentiment_model.predict(data['message'])

	#switch predictions to -1 for negative and +1 for positive to help with unbiased summation
	data['prediction'] = data['prediction'].apply(switch_labels)


	states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DC", "DE", "FL", "GA", 
		"HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", 
		"MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", 
		"NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", 
		"SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]

	states_data = pd.DataFrame()
	states_data['states'] = states
	states_data = states_data.assign(sentiment=0)


	#calculate overall sentiment for each of the states
	print("Calculating Overall Sentiment ...")
	for index, row in data.iterrows():
		loc = states.index(row['state'])
		states_data.at[loc,'sentiment'] = states_data.at[loc,'sentiment'] + row['prediction']


	print ("Creating Heat Map...")

	#create heat map using plotly and save as html for future use using plotly.offline
	#use IFrame to view the created map in browser
	colorscale=[
			[0, 'rgb(31,120,180)'], 
			[0.35, 'rgb(166, 206, 227)'], 
			[0.75, 'rgb(251,154,153)'], 
			[1, 'rgb(227,26,28)']
		   ]
	graph_data = dict(type='choropleth',
			colorscale = colorscale,
			reversescale=True,
			locations = states_data['states'],
			z = states_data['sentiment'],
			locationmode = 'USA-states',
			text = states_data['states'],
			marker = dict(line = dict(color = 'rgb(255,255,255)',width = 2)),
			colorbar = {'title':"Twitter Sentiment"}
			)
	layout = dict(title = 'Twitter Sentiment: ' + keyword,
			  geo = dict(scope='usa'
						)
			 )
	choromap_us = go.Figure(data = [graph_data],layout = layout)
	plotly.offline.plot(choromap_us, filename=keyword + '_sentiment_map.html')
	IFrame(keyword + '_sentiment_map.html', width=950, height=700)


if __name__ == "__main__":
	main()


