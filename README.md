import numpy as np # linear algebra
import seaborn as sns # plotting graphs
import matplotlib.pyplot as plt 
import pandas as pd # data processing
# Importing CSV into pandas dataframe & EDA
African_Financial_Data = pd.read_csv('wk_African_financial_dataset.csv')
African_Financial_Data.head()
African_Financial_Data.describe()
African_Financial_Data.info()
print(African_Financial_Data.shape)
African_Financial_Data.isna().sum()
# ANALYSING DATA - grouping & Sorting 
df = African_Financial_Data.groupby(['country']).count()['systemic_crisis'].sort_values(ascending=False)
df
# ANALYSING DATA - grouping & Sorting 
Fd = African_Financial_Data.groupby('country').agg({'inflation_crises':'sum'}).sort_values('inflation_crises', ascending = False)
Fd
# ANALYSING DATA - Indexing 
for i, row in African_Financial_Data.iloc[:20].iterrows():
    print(f"Index:{i}")
    print(f"{row['country']}")
    
    # ANALYSING DATA - Drop Duplicates
drop_duplicates = African_Financial_Data.drop_duplicates(subset=['country'])
print(drop_duplicates.shape[0])
# ANALYSING DATA - Drop missing rows
drop_rows = African_Financial_Data.dropna()
drop_rows
# ANALYSING DATA - Looping, iterrows
#finding where Egypt features in countries
Egypt_location = African_Financial_Data[African_Financial_Data["country"].str.contains("Egypt")]
for i in Egypt_location:
    print(i)
    # ANALYSING DATA - Looping, iterrows
count = 0
for Egypt in African_Financial_Data['country']:
    if (Egypt == 'Egypt'):
        count += 1
print(count, 'number of times Egypt appeared in the African financial data')
# ANALYSING DATA - Merge Data Frames
countries_world= pd.read_csv('countries of the world.csv')
countries_world.head()
# ANALYSING DATA - Showing the merging of two dataframes
Merge_African_and_World = African_Financial_Data.merge(countries_world,left_on='country',right_on='Country',how="outer")
Merge_African_and_World.head()
# PYTHON - custom function to create reusable code
#function to check number of rows and columns
def rc_myfunc():
    print("The financial_data dataset has " + str(African_Financial_Data.shape[0]) + " rows" +" and " +str(African_Financial_Data.shape[1]) + " columns.")
rc_myfunc()
# PYTHON - Use Of Numpy
#Showing Numpy - Calculating Population Density 
np_Population= np.array(countries_world["Population"]) 
np_Area= np.array(countries_world["Area (sq. mi.)"])
countries_world['Population Density'] =(np_Population/np_Area)
countries_world['Population Density']
# PYTHON - #creating a list
African_country_list = ['Algeria','Angola','Central African Republic','Ivory Coast','Egypt','Kenya','Mauritius',
'Morocco','Nigeria','South Africa','Tunisia','Zambia','Zimbabwe']
print(African_country_list[0])
print(African_country_list[2])
print(len(African_country_list))
# VISUALIZE - Two charts using Matplotlib
#heatmap
plt.figure(figsize=(10,10))
sns.heatmap(African_Financial_Data.corr(),annot=True)
# VISUALIZE - Chart using Matplotlib
sns.set(style='darkgrid')
cols=['systemic_crisis','domestic_debt_in_default','sovereign_external_debt_default','currency_crises','inflation_crises','banking_crisis']
plt.figure(figsize=(20,20))
count=1
for col in cols:
    plt.subplot(3,2,count)
    count+=1
    sns.countplot(y=African_Financial_Data.country,hue=African_Financial_Data[col],palette='rocket')
    plt.legend(loc=0)
    plt.title(col)
plt.tight_layout()
plt.show()
# GENERATE VALUABLE INSIGHTS - 1 Systemic Crisis
fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(African_Financial_Data['country'],hue=African_Financial_Data['systemic_crisis'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)
systemic = African_Financial_Data[['year','country', 'systemic_crisis', 'exch_usd', 'banking_crisis']]
systemic = systemic[(systemic['country'] == 'Central African Republic') | (systemic['country']=='Kenya') | (systemic['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in systemic.country.unique():
    plt.subplot(len(systemic.country.unique()),1,count)
    subset = systemic[(systemic['country'] == country)]
    sns.lineplot(subset['year'],subset['systemic_crisis'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Systemic Crisis/Banking Crisis')
    plt.title(country)
    count+=1
    
    # GENERATE VALUABLE INSIGHTS - 2. Sovereign Domestic Debt Default

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(African_Financial_Data['country'],hue=African_Financial_Data['domestic_debt_in_default'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)
sovereign = African_Financial_Data[['year','country', 'domestic_debt_in_default', 'exch_usd', 'banking_crisis']]
sovereign = sovereign[(sovereign['country'] == 'Angola') | (sovereign['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in sovereign.country.unique():
    plt.subplot(len(sovereign.country.unique()),1,count)
    subset = sovereign[(sovereign['country'] == country)]
    sns.lineplot(subset['year'],subset['domestic_debt_in_default'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Sovereign Domestic Debt Defaults/Banking Crisis')
    plt.title(country)
    count+=1
    
  # GENERATE VALUABLE INSIGHTS - 3. Sovereign External Debt Default

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(African_Financial_Data['country'],hue=African_Financial_Data['sovereign_external_debt_default'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)
sovereign_ext = African_Financial_Data[['year','country', 'sovereign_external_debt_default', 'exch_usd', 'banking_crisis']]
sovereign_ext = sovereign_ext[(sovereign_ext['country'] == 'Central African Republic') | (sovereign_ext['country'] == 'Ivory Coast') | (sovereign_ext['country']=='Zimbabwe') ]
plt.figure(figsize=(20,15))
count = 1

for country in sovereign_ext.country.unique():
    plt.subplot(len(sovereign_ext.country.unique()),1,count)
    subset = sovereign_ext[(sovereign_ext['country'] == country)]
    sns.lineplot(subset['year'],subset['sovereign_external_debt_default'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Sovereign Ext Debt Defaults/Banking Crisis')
    plt.title(country)
    count+=1
   
   # GENERATE VALUABLE INSIGHTS - 4. Currency Crisis

fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(African_Financial_Data['country'],hue=African_Financial_Data['currency_crises'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)
curr = African_Financial_Data[['year','country', 'currency_crises', 'exch_usd', 'banking_crisis']]
curr = curr[(curr['country'] == 'Angola') | (curr['country'] == 'Zambia') | (curr['country']=='Zimbabwe') ]
curr = curr.replace(to_replace=2, value=1, regex=False)

plt.figure(figsize=(20,15))
count = 1

for country in curr.country.unique():
    plt.subplot(len(curr.country.unique()),1,count)
    subset = curr[(curr['country'] == country)]
    sns.lineplot(subset['year'],subset['currency_crises'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Currency Crisis/Banking Crisis')
    plt.title(country)
    count+=1
    
  # GENERATE VALUABLE INSIGHTS - 5. Inflation Crisis
  fig,ax = plt.subplots(figsize=(20,10))
sns.countplot(African_Financial_Data['country'],hue=African_Financial_Data['inflation_crises'],ax=ax)
plt.xlabel('Countries')
plt.ylabel('Counts')
plt.xticks(rotation=45)
infla = African_Financial_Data[['year','country', 'inflation_crises', 'inflation_annual_cpi', 'banking_crisis']]
infla = infla[(infla['country'] == 'Angola') | (infla['country'] == 'Zambia') | (infla['country']=='Zimbabwe') ]
infla = infla.replace(to_replace=2, value=1, regex=False)

plt.figure(figsize=(20,15))
count = 1

for country in infla.country.unique():
    plt.subplot(len(infla.country.unique()),1,count)
    subset = infla[(infla['country'] == country)]
    sns.lineplot(subset['year'],subset['inflation_crises'],ci=None)
    plt.scatter(subset['year'],subset["banking_crisis"], color='coral', label='Banking Crisis')
    plt.subplots_adjust(hspace=0.6)
    plt.xlabel('Years')
    plt.ylabel('Inflation Crisis/Banking Crisis')
    plt.title(country)
    count+=1
    
    ##The rise in the annual CPI for the countries co-incide around the same time period when the country was 
##facing a banking crisis.











