import pandas as pd
df=pd.read_csv("ipl2017.csv")
df.head()
df.info()
df.describe()


# Handling Missing Values

df.isnull()


# Dropping Down The Unnecessary Columns

y=df["total"]
x=df.drop(['total','batsman','bowler','mid','date'],axis=1)
x.head()
x.dtypes
x=pd.get_dummies(x,columns=['bat_team','bowl_team','venue'],drop_first=True)
x.head()
x.shape          # Checking The Shape Of x
type(x)          #Checking The Type Of x 
x.columns


# Converting Categorical Strings Columns To Numerical Columns, By Using One-Hot Encoding

from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df.venue.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
ohe.fit_transform(df.venue.values.reshape(-1, 1)).toarray()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df.bat_team.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
ohe.fit_transform(df.bat_team.values.reshape(-1, 1)).toarray()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df.bowl_team.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
ohe.fit_transform(df.bowl_team.values.reshape(-1, 1)).toarray()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df.batsman.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
ohe.fit_transform(df.batsman.values.reshape(-1, 1)).toarray()


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder()
ohe.fit_transform(df.bowler.values.reshape(-1, 1)).toarray()
ohe = OneHotEncoder(drop='first')
ohe.fit_transform(df.bowler.values.reshape(-1, 1)).toarray()


# Performing Split

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42)


# Performing Feature Scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)


# Building A Model On "total" Columns, Using A RandomForestRegressor

from sklearn.ensemble import RandomForestRegressor
forest=RandomForestRegressor()
forest.fit(x_train,y_train)


# Calculating The Score

forest.score(x_test,y_test)
forest.score(x_train,y_train)


# Predicting On A New Set Of Features

match={ 'runs':[146,42], 'wickets':[2,2], 'overs':[12.4,6.1],'runs_last_5':[47,39], 'wickets_last_5':[1,2],
       'striker':[52,3], 'non-striker':[6,1],
       'bat_team_Deccan Chargers':[0,0], 'bat_team_Delhi Daredevils':[0,0],
       'bat_team_Gujarat Lions'  :[0,1], 'bat_team_Kings XI Punjab':[1,0],
       'bat_team_Kochi Tuskers Kerala':[0,0], 'bat_team_Kolkata Knight Riders':[0,0],
       'bat_team_Mumbai Indians':[0,1], 'bat_team_Pune Warriors':[0,0],
       'bat_team_Rajasthan Royals':[0,0], 'bat_team_Rising Pune Supergiant':[0,0],
       'bat_team_Rising Pune Supergiants':[0,0],
       'bat_team_Royal Challengers Bangalore':[0,0], 'bat_team_Sunrisers Hyderabad':[0,1],
       'bowl_team_Deccan Chargers':[0,0], 'bowl_team_Delhi Daredevils':[0,0],
       'bowl_team_Gujarat Lions':[0,0], 'bowl_team_Kings XI Punjab':[0,1],
       'bowl_team_Kochi Tuskers Kerala':[0,0], 'bowl_team_Kolkata Knight Riders':[0,0],
       'bowl_team_Mumbai Indians':[1,0], 'bowl_team_Pune Warriors':[0,0],
       'bowl_team_Rajasthan Royals':[0,0], 'bowl_team_Rising Pune Supergiant':[0,0],
       'bowl_team_Rising Pune Supergiants':[0,0],
       'bowl_team_Royal Challengers Bangalore':[0,0],
       'bowl_team_Sunrisers Hyderabad':[0,0],'venue_Brabourne Stadium':[0,0],
       'venue_Buffalo Park':[0,0], 'venue_De Beers Diamond Oval':[0,0],
       'venue_Dr DY Patil Sports Academy':[0,0],
       'venue_Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':[0,0],
       'venue_Dubai International Cricket Stadium':[0,0], 'venue_Eden Gardens':[0,1],
       'venue_Feroz Shah Kotla':[0,0], 'venue_Green Park':[0,0],
       'venue_Himachal Pradesh Cricket Association Stadium':[0,0],
       'venue_Holkar Cricket Stadium':[0,0],
       'venue_JSCA International Stadium Complex':[0,0], 'venue_Kingsmead':[0,0],
       'venue_M Chinnaswamy Stadium':[0,0], 'venue_MA Chidambaram Stadium, Chepauk':[0,0],
       'venue_Maharashtra Cricket Association Stadium':[0,0], 'venue_Nehru Stadium':[0,0],
       'venue_New Wanderers Stadium':[0,0], 'venue_Newlands':[0,0],
       'venue_OUTsurance Oval':[0,0,],
       'venue_Punjab Cricket Association IS Bindra Stadium, Mohali':[0,0],
       'venue_Punjab Cricket Association Stadium, Mohali':[0,0],
       'venue_Rajiv Gandhi International Stadium, Uppal':[0,1],
       'venue_Sardar Patel Stadium, Motera':[0,0],
       'venue_Saurashtra Cricket Association Stadium':[0,0],
       'venue_Sawai Mansingh Stadium':[0,0],
       'venue_Shaheed Veer Narayan Singh International Stadium':[0,0],
       'venue_Sharjah Cricket Stadium':[0,0], 'venue_Sheikh Zayed Stadium':[0,0],
       'venue_St Georges Park':[0,0], 'venue_Subrata Roy Sahara Stadium':[0,0],
       'venue_SuperSport Park':[0,0],
       'venue_Vidarbha Cricket Association Stadium, Jamtha':[0,0],
       'venue_Wankhede Stadium':[0,0]
       }

data=pd.DataFrame(match)
type(data)
data
forest.predict(data)


# Data Visualization

import matplotlib.pyplot as plt
df.hist()
import seaborn as sns
sns.pairplot(df)