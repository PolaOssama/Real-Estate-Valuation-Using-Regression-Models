import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression , Ridge ,LassoCV
from sklearn.ensemble import RandomForestRegressor
from yellowbrick.regressor import ResidualsPlot ,PredictionError
from sklearn.metrics import  r2_score
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import  StandardScaler 
import math
from sklearn.preprocessing import PolynomialFeatures
df = pd.read_excel('Real estate valuation data set.xlsx')
df.head()   # return fist 5 rows
df.shape    # retun dataset dimension
df.info()   # check dataType of each column & check if it conains null

################## pairplot #################
sns.pairplot(df)                     
########## pairplot two pair vaiable with every feature  no. of feature C 2 ##############
df_new = df 
df_new.drop('No', inplace=True, axis=1)    #Remove The First Coulumn "NO"

# Rename the columns (remove X1..X2..X3.. and Y)
df_new.columns = ['transaction date', 'house age', 'distance to the nearest MRT station', 'number of convenience stores', 'latitude', 'longitude', 'house price of unit area']

# Split the data into train and test
trainData, testData = train_test_split(df_new, test_size=0.2, random_state=1)

########################################################################################
#################### Visualize Data Before Removing outlier  ###########################
########################################################################################
      
#**************** Histogram *******************
fig = plt.figure(figsize=(20,20))
for index,col in enumerate(trainData):
    plt.subplot(6,3,index+1)
    sns.histplot(trainData.loc[:,col].dropna(), kde=True, stat="density", linewidth=0.5);
fig.tight_layout(pad=1.0);
fig.suptitle('Data Before Removing outlier', fontsize=16).set_position([.5, 1.02])
   
#**************** Box Plot *******************
fig = plt.figure(figsize=(14,15))
for index,col in enumerate(trainData):
    plt.subplot(6,3,index+1)
    sns.boxplot(y=col, data=trainData.dropna())
    plt.grid()
fig.tight_layout(pad=1.0)
fig.suptitle('Data Before Removing outlier', fontsize=16).set_position([.5, 1.02]);

#//////////////////////////////_\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#
#///////////////////////// Remove Outlier \\\\\\\\\\\\\\\\\\\\\\\\\\#
#//////////////////////////////_\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\#

trainData = trainData[trainData['longitude']>121.50]
trainData = trainData[trainData['house price of unit area']<80]
trainData = trainData[trainData['distance to the nearest MRT station']<3000]

########################################################################################
#################### Visualize Data After Removing outlier  ###########################
########################################################################################

################## Heat MAP #################
sns.heatmap(trainData.corr(), annot=True,cmap='RdPu')


fig, ax = plt.subplots(2, 3, figsize=(20, 9))
ax = ax.flatten()
sns.set()
sns.lineplot(data=trainData, x="transaction date", y="house price of unit area", ax=ax[0])
ax[0].set_title("Price of Unit Area vs. Transaction Date")

sns.lineplot(data=trainData, x="house age", y="house price of unit area", ax=ax[1])
ax[1].set_title("Price of Unit Area vs. House Age")

sns.lineplot(data=trainData, x="distance to the nearest MRT station", y="house price of unit area", ax=ax[2])
ax[2].set_title("Price of Unit Area vs. Distance to the nearest MRT station")

sns.lineplot(data=trainData, x="number of convenience stores", y="house price of unit area", ax=ax[3])
ax[3].set_title("Price of Unit Area vs. no. of Convenience Stores")

sns.lineplot(data=trainData, x="latitude", y="house price of unit area", ax=ax[4])
ax[4].set_title("Price of Unit Area vs. Latitude")

sns.lineplot(data=trainData, x="longitude",y="house price of unit area", ax=ax[5])
ax[5].set_title("Price of Unit Area vs. Longitude")

# set the spacing between subplots
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.4, 
                    hspace=0.4)
fig.suptitle('Data After Removing outlier', fontsize=16).set_position([.5, 1.02])
plt.show();

                          #**************** Box Plot *******************
fig = plt.figure(figsize=(14,15))
for index,col in enumerate(trainData):
    plt.subplot(6,3,index+1)
    sns.boxplot(y=col, data=trainData.dropna())
    plt.grid()
fig.tight_layout(pad=1.0)
fig.suptitle('Data After Removing Outlier', fontsize=16).set_position([.5, 1.02]);

                         #**************** Histogram *******************
fig = plt.figure(figsize=(16,16))
for index,col in enumerate(trainData):
    plt.subplot(6,3,index+1)
    sns.histplot(trainData.loc[:,col].dropna(), kde=True, stat="density", linewidth=0.5);
fig.tight_layout(pad=1.0);
fig.suptitle('Data After Removing Outlier', fontsize=16).set_position([.5, 1.02]);

#########################3 Concat to perform preprocessing and scaling #################
df_last = [trainData, testData]
data = pd.concat(df_last)

########################################################################################
############################# Change Date format  ######################################
########################################################################################
def ChangeDateFormat(data):
    # STEP 1: Convert transaction date to day, month and year columns
    # Create date column with `transaction date` as a date
    data['date'] = pd.to_datetime(data['transaction date'], format='%Y')  # EX 2012-0-0

    # Create year column
    data['year'] = pd.DatetimeIndex(data['date']).year  #2012

    # Create month column by extracting the decimal part of `transaction date` and multiplying it by 12
    data['month'], data['year isolate'] = data['transaction date'].apply(lambda x: math.modf(x)).str  # 92
    data['month'] = data['month']*12

    # Create day column by extracting the decimal part of int
    data['day'], data['month'] = data['month'].apply(lambda x: math.modf(x)).str

    # Convert month to int
    data['month'] = (data['month']).astype(int)

    # Drop unnecessary columns
    data = data.drop(['transaction date', 'date', 'year isolate'], axis=1, inplace=True)

ChangeDateFormat(data)

########################################################################################
#################### Split Data & Train Linear Regression Model ########################
########################################################################################
X=data.drop('house price of unit area',axis=1)
y=data['house price of unit area']


################### Z SCORING ######################  z = x-M/s

transformer = StandardScaler().fit(X)
trans_Data = transformer.transform(X)   

polynomial_converter = PolynomialFeatures(degree=2, include_bias=False)

# Fit and transform
featureEngineering = polynomial_converter.fit(trans_Data)
featureEngineering = polynomial_converter.transform(trans_Data)
featureEngineering.shape

# VAlIDATION IF NEEDED 
X_train,X_test,y_train,y_test = train_test_split(featureEngineering,y,test_size=0.2,random_state=1)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


#X_train,X_test,y_train,y_test = train_test_split(poly_features,y,test_size=0.45,random_state=1)

LinearRegressionModel = LinearRegression()
LinearRegressionModel.fit(X_train,y_train)
 

########################################################################################
#################### Linear Regreession Model Evaluation ###############################
########################################################################################
# Predict on  test data
pred_test = LinearRegressionModel.predict(X_test)   

pd.DataFrame({'y_test':y_test,'pred_test':pred_test,'Residuals':(y_test-pred_test)}).head(8)



# Predict on train data
pred_train = LinearRegressionModel.predict(X_train)

r2_trainn = r2_score(y_train,pred_train)                        # R square
MAE_train = metrics.mean_absolute_error(y_train, pred_train)    # Mean Absolute Error
MSE_train = metrics.mean_squared_error(y_train, pred_train)     # Mean Squared Error
RMSE_train= np.sqrt(MSE_train)                                  # Root Mean Squared Error
                               
# Predict on val data
pred_val = LinearRegressionModel.predict(X_val)

R2_val = r2_score(y_val, pred_val)                           # R square
MAE_val = metrics.mean_absolute_error(y_val, pred_val)        # Mean Absolute Error
MSE_val = metrics.mean_squared_error(y_val , pred_val)        # Mean Squared Error
RMSE_val = np.sqrt(MSE_val)                                   # Root Mean Squared Error



pd.DataFrame({'Validation':  [R2_val, MSE_val, RMSE_val, MAE_val],
               'Training': [r2_trainn, MSE_train, RMSE_train, MAE_train],
             },
              index=['R2', 'MSE', 'RMSE', 'MAE'])

R2_test = r2_score(y_test, pred_test)                        # R square
MAE_test = metrics.mean_absolute_error(y_test, pred_test)    # Mean Absolute Error
MSE_test = metrics.mean_squared_error(y_test, pred_test)     # Mean Squared Error
RMSE_test = np.sqrt(MSE_test)                                # Root Mean Squared Error


#*************************** Linear Regressor ResidualsPlot ************************************

visualizer = ResidualsPlot(LinearRegressionModel, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();

#*************************** Visualize Original vs Predicted ***********************************

x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test, label="predicted")
plt.title("Linear test and predicted data")
plt.legend()
plt.show();


visualizer = PredictionError(LinearRegressionModel)
visualizer.fit(X_train, y_train)  
visualizer.score(X_test, y_test)  
visualizer.poof()

########################################################################################
########################### Train XGBRegressor Model ###################################
########################################################################################

model_xgbr = XGBRegressor(objective ='reg:squarederror',n_estimators = 10, seed = 0, max_depth = 3)
model_xgbr.fit(X_train, y_train)

########################################################################################
############### XGBRegressor Model Evaluation "Gradient Boosting Decision Tree" ######## 
################ we build a tree with the goal of predicting the residuals #############
########################################################################################
pred_test_xgbr = model_xgbr.predict(X_test)

r2_test_xgbr = r2_score(y_test, pred_test_xgbr)                      # R square  Its the diffrence between var(mean) - var(line) / var(mean)
MSE_test_xgbr = metrics.mean_squared_error(y_test, pred_test_xgbr)   # Mean Squared Error
RMSE_test_xgbr = np.sqrt(MSE_test_xgbr)                              # Root Mean Squared Error
MAE_test_xgbr = metrics.mean_squared_error(y_test, pred_test_xgbr)   # Mean Absolute Error

#****************** XGBRegressor ResidualsPlot *************************
visualizer = ResidualsPlot(model_xgbr, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();
#****************** Visualize Original vs Predicted *************************
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test_xgbr, label="predicted")
plt.title("XGBRegressor test and predicted data")
plt.legend()
plt.show();

###########################################################################################################################
######################### Train LASSO Model  "Least Absolute Shrinkage and Selection Operator" ############################
################# Shrinkage is where data values are shrunk towards a central point, like the mean ########################
###########################################################################################################################
model_lasso_cv = LassoCV(eps=0.01, n_alphas=100, cv=10, max_iter=10000)
#model_lasso_cv = LassoCV(eps=0.01, n_alphas=100, cv=10, max_iter=10000)  # alpha = 0.1 / 10 /100  
model_lasso_cv.fit(X_train, y_train)

pred_test_lasso = model_lasso_cv.predict(X_test)

########################################################################################
########################## LASSO Model Evaluation ######################################
########################################################################################
r2_test_lasso = r2_score(y_test, pred_test_lasso)
mse_test_lasso =metrics.mean_squared_error(y_test, pred_test_lasso)
rmse_test_lasso = np.sqrt(mse_test_lasso)
mae_test_lasso = metrics.mean_absolute_error(y_test, pred_test_lasso)

#****************** LASSO ResidualsPlot *************************
visualizer = ResidualsPlot(model_lasso_cv, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();
#****************** Visualize Original vs Predicted *************************
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test_lasso, label="predicted")
plt.title("LASSO test and predicted data")
plt.legend()
plt.show();

########################################################################################
########################### Train DecisionTreeRegressor Model #########################
#######################################################################################
regressor = DecisionTreeRegressor(random_state = 50 , max_depth = 3) 
regressor.fit(X_train, y_train)
pred_test_regressor = regressor.predict(X_test)

########################################################################################
#################### DecisionTreeRegressor Model Evaluation ############################
########################################################################################

r2_test_reg = r2_score(y_test, pred_test_regressor)
mse_test_reg =metrics.mean_squared_error(y_test, pred_test_regressor)
rmse_test_reg = np.sqrt(mse_test_reg)
mae_test_reg = metrics.mean_absolute_error(y_test, pred_test_regressor)

#****************** DecisionTreeRegressor ResidualsPlot *************************
visualizer = ResidualsPlot(regressor, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();
#****************** Visualize Original vs Predicted *************************
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test_regressor, label="predicted")
plt.title("DecisionTreeRegressor test and predicted data")
plt.legend()
plt.show();

########################################################################################
########################### Train Ridge Model ##########################################
########################################################################################
ridge_model = Ridge(alpha = 0.1)
ridge_model.fit(X_train, y_train)

pred_test_ridge = ridge_model.predict(X_test)

########################################################################################
####################  Ridge Model Evaluation ############################################
########################################################################################
r2_test_ridge = r2_score(y_test, pred_test_ridge)
mse_test_ridge =metrics.mean_squared_error(y_test, pred_test_ridge)
rmse_test_ridge = np.sqrt(mse_test_ridge)
mae_test_ridge = metrics.mean_absolute_error(y_test, pred_test_ridge)

#****************** Ridge ResidualsPlot *************************
visualizer = ResidualsPlot(ridge_model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();
#****************** Visualize Original vs Predicted *************************
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test_regressor, label="predicted")
plt.title("Ridge test and predicted data")
plt.legend()
plt.show();

###############################################################################################################
############################################ Train RandomForest Model #########################################
###############################################################################################################
RandomForestRegressor_model = RandomForestRegressor(max_depth=5 ,n_estimators = 100)
RandomForestRegressor_model.fit(X_train, y_train)

pred_test_randomForest = RandomForestRegressor_model.predict(X_test)

########################################################################################
####################  RandomForest Model Evaluation ####################################
########################################################################################
r2_test_ranFor = r2_score(y_test, pred_test_randomForest)
mse_test_ranFor =metrics.mean_squared_error(y_test, pred_test_randomForest)
rmse_test_ranFor = np.sqrt(mse_test_ranFor)
mae_test_ranFor = metrics.mean_absolute_error(y_test, pred_test_randomForest)

#****************** Random Forest ResidualsPlot *************************
visualizer = ResidualsPlot(RandomForestRegressor_model, hist=False, qqplot=True)
visualizer.fit(X_train, y_train)
visualizer.score(X_test, y_test)
visualizer.show();
#****************** Visualize Original vs Predicted *************************
x_ax = range(len(y_test))
plt.plot(x_ax, y_test, label="original")
plt.plot(x_ax, pred_test_randomForest, label="predicted")       
plt.title("Random Forest test and predicted data")
plt.legend()
plt.show();

####################################################################################################################################
################################################ ALL Models Evaluation #############################################################
####################################################################################################################################

#****************** R2 Comparison *************************
sns.set()
plt.figure(figsize=(10,6))
models = ['Linear Regression', 'XGBRegressions', 'Random Forest','LassoCV' , 'Descion tree Regresor' , 'Ridge']
r2 = [R2_test, r2_test_xgbr,r2_test_ranFor, r2_test_lasso , r2_test_reg , r2_test_ridge]
ax = sns.barplot(x = models, y = r2, palette='pastel')
ax.bar_label(ax.containers[0])
plt.xlabel('Models')
plt.ylabel('R2 Score')
plt.title('Comparing R2 Score of Models');

#****************** RMSE Comparison *************************
sns.set()
plt.figure(figsize=(10,6))
rmse = [RMSE_test, RMSE_test_xgbr,rmse_test_ranFor ,rmse_test_lasso , rmse_test_reg , rmse_test_ridge]
ax = sns.barplot(x = models, y = rmse, palette = 'pastel')
ax.bar_label(ax.containers[0])
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('Comparing RMSE of Models');