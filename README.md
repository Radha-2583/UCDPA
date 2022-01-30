import numpy as np # linear algebra
import seaborn as sns # plotting graphs
import matplotlib.pyplot as plt 
import pandas as pd # data processing
# Importing CSV into pandas dataframe & EDA
African_Financial_Data = pd.read_csv('wk_African_financial_dataset.csv')
African_Financial_Data.head()
African_Financial_Data.describe()
