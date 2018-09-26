# Stock-Price-Prediction

The code uses the Keras machine learning library to train the network on a stock price dataset (from Google Finance) in order to predict a future price.
It uses Tweepy to retrieve tweets about the stock. Then, employs TextBlob to determine if the majority of the tweets are positive using sentiment analysis. If the majority tweets are positive, it trains a neural net with that data to predict the price for tomorrow.

##Dependencies

* numpy (http://www.numpy.org/)
* tweepy (http://www.tweepy.org)
* csv (https://pypi.python.org/pypi/csv)
* textblob (https://textblob.readthedocs.io/en/dev/)
* keras (https://keras.io)

Install missing dependencies using [pip](https://pip.pypa.io/en/stable/installing/)

##Demo Usage

Once you have your dependencies installed via pip, run the script in terminal via

```
python Prediction.py
```
