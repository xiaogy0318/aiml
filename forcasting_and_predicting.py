# The Linear Regression to predict a stock price in the future

# Precondition
# Python 3.x
# pip install a bunch of things, i.e. sklearn and matplotlib

import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import argparse
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')

parser = argparse.ArgumentParser(description='Machine learning stock prices for BRK_A')
parser.add_argument('text', type=str,
                    help='Input your quandl api key')


args = parser.parse_args()

quandl.ApiConfig.api_key = args.text
df = quandl.get('WIKI/BRK_A')
#df = quandl.get('WIKI/GOOGL')

# This is added to avoid issue like warnings.warn("Numerical issues were encountered "
#df = df.astype(float)

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'

#forecast_out = int(math.ceil(0.01 * len(df))) # for this case it's 100, and the score is about 75%
forecast_out = int(math.ceil(0.01 * len(df))) # for this case it's 10, and the score is about 98%
# It makes sense when shifting is too big, the score would drop.
# Why shifting here? Because if no shifting at all, it's gonna be 100%. Of course it will be, because its the actual data
# In order to get a meaningful score (not 100%), shifting is one way to do it.
# Is it the only way to do it???

print(forecast_out)

df['label'] = df[forecast_col].shift(-forecast_out)
df.fillna(-99999, inplace=True)


df.dropna(inplace=True)
y = np.array(df['label'])

print(df)

# Prepare to predict

X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X) # This step must be before the downsizing of X, otherwise it'd plot a flat line.
X_lately = X[-forecast_out:]
X = X[:-forecast_out]

y = y[:-forecast_out] # this step seems missing in the original YT video. Without it you get a mismatch between X and y

#X = preprocessing.scale(X)
#y = preprocessing.scale(y) # Not sure if this is needed or not

# Split the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train data
clf = LinearRegression()
# clf = svm.SVR() # this is just to show you can plug in a different "engine" to predict. In this case svm is very bad (score is negative)
clf.fit(X_train, y_train)

# Score the test data per trained data
score = clf.score(X_test, y_test)
print(score)

# Predict new stock prices
forcast_set = clf.predict(X_lately)

print(forcast_set, score, forecast_out)

# Plot the existing data and the forcast
df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forcast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()