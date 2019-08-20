# H8rs-Gonna-H8
Geographic Tweet Sentiment Analysis

Trained a Naive Bayes model to predict tweet sentiment and then developed a program that allows a user to input a phrase to search the Twitter API against and then displays overall state sentiment using a heat map. 

The training and test datasets can be found at http://help.sentiment140.com/home. The training dataset contains 1.6 million pre-labelled tweets (4 being a positive tweet and 0 identifying a negative tweet). The test dataset contains roughly 500 tweets. 

Source Files Included (Please see comments in individual files for detailed explanations): 

sentiment_analysis_model_generator.py: Script that actually trains the model. Can take a significant amount of time to run!

sentiment_analysis_model_evaluation.py: Uses sklearn evaluation metrics against the trained model.

sentiment_analysis.py: Usage: python3 sentiment_analysis.py [phrase]. Allows user to input a phrase and then fetches relevant tweets using the tweepy Twitter API wrapper. Uses the model to predict sentiment, aggregates sentiment by state and produces a heat map. 


Additional Files:

sentiment_analysis_model.pkl: Pre-trained model for ease of use.

model_evaluation: Output of the model evaluation metrics.

model_generation_summary: Summarizes the metrics of intermediate models based each combination of grid search parameters.

NOTE: Before running sentiment_analysis.py you must use personal consumer and access tokens from the Twitter API. See code for exact location.
