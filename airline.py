from random import random
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import sklearn
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

# GETTING THE COLUMN NAMES
labels = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2018.csv",nrows=1)

# READING ALL DATA JUST EXCLUDING A CLASS LABEL (ARR_DELAY)
data = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2018.csv",encoding='utf-8',usecols=[col for col in labels if col!='ARR_DELAY'])
catCols = [col for col in data.columns if data[col].dtype=='O']
data = data.fillna(data.mode().iloc[0])
data1 = data.dropna(axis=1,how='all')


# WE WILL PREDICT ARRIVAL DELAY SO WE USE IT AS OUR CLASS LABEL (ARR_DELAY)
y = pd.read_csv("/home/knez/Desktop/vsc_code/Airline data/2018.csv",encoding='utf-8',usecols=["ARR_DELAY"])
y = y.fillna(y.mode().iloc[0])
y = y.values
# using label encoder to convert categorical attributes to numerical
le = LabelEncoder()

for i in range(len(data1.columns)):
    tmp = data1.columns[i]
    if tmp in catCols:
        data1[tmp]=le.fit_transform(data1[tmp])



# selecting only 20% of our data because of my CPU
x = data1.values
X,X_,y,y_ = train_test_split(x,y,test_size=0.8,stratify=data1["FL_DATE"].values)

del X_,y_


# standardize data using StandardScaler()
sc = StandardScaler()
X_std = sc.fit_transform(X)


from itertools import combinations

# DEFINIG SBS (sequential backward selection) ALGORITHM DO SELECT ATTRIBUTES WHICH ARE MOST IMPORTANT FOR PREDICTIONS

class SBS():
    def __init__(self,estimator,k_features,scoring=r2_score,test_size=0.3,random_state=1):
        self.scoring = scoring # criterium we use for select atributes
        self.estimator=estimator # estimator we use
        self.k_features=k_features # number of all atributes
        self.test_size=test_size # test size
        self.random_state=random_state 
    
    def fit(self,X,y):
        # splitting data into training and test
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=self.test_size,random_state=self.random_state) 

        dim = X_train.shape[1] # determine num of attributes
        self.indices_ = tuple(range(dim)) # making tuple for attributes
        self.subsets_=[self.indices_] # subsets of attributes
        score = self._calc_score(X_train,y_train,X_test,y_test,self.indices_) # score calculating

        self.scores_ = [score]

        while dim > self.k_features: # while dim is bigger than num of attributes
            scores = []
            subsets =[]

            for p in combinations(self.indices_,r = dim-1): # getting all combinations without one attribute,calculating and getting best results
                score = self._calc_score(X_train,y_train,X_test,y_test,p)
                
                scores.append(score)
                subsets.append(p)
            
            best = np.argmax(scores) # determine best result
            self.indices_ = subsets[best] # selecting best by index
            self.subsets_.append(self.indices_) 
            dim -= 1

            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]

        return self

    def transform(self,X):
        return X[:,self.indices_] # returnig only selected attributes, one attribute smaller than start 
    
    def _calc_score(self,X_train,y_train,X_test,y_test,indices): # this is for determine which attribute should be removed as it dont impact much for our prediction
        self.estimator.fit(X_train[:,indices],y_train)
        y_pred = self.estimator.predict(X_test[:,indices])
        score = self.scoring(y_test,y_pred)
        return score
    
            


# DIVIDING DATA IN TRAINING AND TEST SETS
X_train,X_test,y_train,y_test = train_test_split(X_std,y,test_size=0.3,random_state=1)

lr = LinearRegression(fit_intercept=True,n_jobs=-1) # linear reggresion model
sbs = SBS(lr,k_features=1) # using sbs while we dont have just one attribute
sbs.fit(X_train,y_train) # fitting sbs

k_feat = [len(k) for k in sbs.subsets_] # determine number of subsets

# ploting accuracy depending of attribute number
plt.plot(k_feat,sbs.scores_,marker='o',color='red')
plt.ylabel("Accuracy")
plt.xlabel("Number of attributes")
plt.grid()
plt.title("Determine accuracy of predicting based on number of attributes")
plt.xticks(np.arange(1,len(k_feat)+1,1))
plt.tight_layout()
plt.show()

# selecting 10 attributes 
feat13 = list(sbs.subsets_[13]) # 13 because we have total 26 attributes so on 13, 13 attributes left
print(data.columns[1:][feat13]) # getting their names


# fit data linear regression
lr.fit(X_train[:,feat13],y_train)
y_pred_train = lr.predict(X_train[:,feat13])
# calculate training accuracy score using r2_score
print("Training accuracy: %.3f" %r2_score(y_train,y_pred_train))
# calculate test accuracy score using r2_score
y_pred_test = lr.predict(X_test[:,feat13])
print("Test accuracy: %.3f" %r2_score(y_test,y_pred_test))


