import pandas as pd
import quandl
import math

#Get all the columns of Google's stock
#Basics:
#features: input, labels: output
df = quandl.get('WIKI/GOOGL')

#Get only the necessary columns
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]

#Generate a 'high (minus) low percentage' column
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) / df['Adj. Close'] * 100

#Generate a 'percentage change' column
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

#Further filter the columns
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

#The column name which is to be forecasted
forecast_col = 'Adj. Close'

#Fill the NA(Not Available) columns with -99999
df.fillna(-99999, inplace=True)

#How far into the future do you wish to go? This will sort it out
#Fiddle around with the multiplier of 'len(df)' to change the choice
forecast_out = int(math.ceil(0.01*len(df)))

#Make a 'label' column which is the column which is forecasted
df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)

print(df.tail())