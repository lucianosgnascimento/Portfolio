#importing libraries
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

def teste():
    print('sucesso')

############################### Analise exploratoria ################################



def importance(model):
    #função para plotar as importancias e devolver uma lista com as importancias de cada feature
    from matplotlib import pyplot
    importance = model.feature_importances_
    # summarize feature importance
    for i,v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i,v))
        
    # plot feature importance
    pyplot.bar([x for x in range(len(importance))], importance)
    pyplot.show()
    return importance


def plotCorrelation(df):
    #Using Pearson Correlation
    plt.figure(figsize=(12,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()  


def relevantFeatures(df,targetName,limit):
    #Correlation with output variable
    cor = df.corr()
    cor_target = abs(cor[targetName])
    #Selecting highly correlated features
    relevant_features = cor_target[cor_target>limit]
    return relevant_features

def featurePerformance(model):
    print(model.pvalues)

############################## DataPrep ######################################


def removerCols(df,colunas):
    df = df.drop(df.columns[colunas],axis = 1)
    return df

def removerCol(df,nomeColuna):
    df = df.drop([nomeColuna],axis=1)
    return df

def typeEncoder(df,tipo):
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()

    #x.dtypes
    converts = []
    contador = 0
    for i in df.dtypes:
        if str(tipo) == str(i):
            converts.append(contador)
        contador = contador + 1
    converts
    for i in converts:
        df.iloc[:,i] = le.fit_transform(df.iloc[:,i])

def listNA(df):
    print(df.isna().sum().sort_values())

def removeLinhaNA(df,colName):
    return df[df[colName].notna()]



##############################  Avaliacao de modelos #########################


def regScore(model,x_test,y_test,pred):
    from sklearn import metrics

    r2_score = model.score(x_test,y_test)
    print("R2:{:.3f}".format(r2_score))
    print("MSE:{:.3f}".format((metrics.mean_squared_error(y_test,pred))))
    print("MAE:{:.3f}".format((metrics.mean_absolute_error(y_test,pred))))



############################ EXEMPLOS ##########################


def RandomSearchEx():
    print('''
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV


    model = RandomForestRegressor()
    param_grid = {'n_estimators':[1,10,100,200],
                    'max_depth': [5,10,15],
                    'min_samples_split':[2,3,5,8]}

    score = scoring=['neg_mean_squared_error','r2']

    grid_cv_ridge = RandomizedSearchCV(model,param_grid,scoring=score,cv=10,verbose=3,refit='neg_mean_squared_error')

    grid_cv_ridge.fit(x_train,y_train)
    print("RefitScore::{}".format(grid_cv_ridge.best_score_))
    print("hiperparametro melhor::{}".format(grid_cv_ridge.best_params_))

    pd.DataFrame(data = grid_cv_ridge.cv_results_).head(3)

    best_model = grid_cv_ridge.best_estimator_

    print(best_model.get_params())

    print(grid_cv_ridge.best_score_)
    ''')

def LassoFeatureSelectionEX():
    print('''
    search.fit(X_train,y_train)
    #The best value for α is:

    search.best_params_
    # {'model__alpha': 1.2000000000000002}
    #Now, we have to get the values of the coefficients of Lasso regression.

    coefficients = search.best_estimator_.named_steps['model'].coef_
    #The importance of a feature is the absolute value of its coefficient, so:

    importance = np.abs(coefficients)
    #Let’s take a look at the importance:


    #As we can see, there are 3 features with 0 importance. Those features have been discarded by our model.

    #The features that survived the Lasso regression are:

    np.array(features)[importance > 0]
    # array(['age', 'sex', 'bmi', 'bp', 's1', 's3', 's5'], dtype='<U3')
    #While the 3 discarded features are:

    np.array(features)[importance == 0]
    # array(['s2', 's4', 's6'], dtype='<U3')
    
    ''')










