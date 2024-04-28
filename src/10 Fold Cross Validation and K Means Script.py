# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:30:18 2024
Main Python Script for 10 Fold Cross Validation, K Means and ANOVA analysis

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn import tree
from sklearn import neighbors
from sklearn import ensemble
import matplotlib.pyplot as plt
from scipy.stats import f_oneway
import seaborn as sns
from seaborn_qqplot import pplot
from scipy.stats import exponnorm

# Define the function that will concatenate all the K Folds into one

def K_Fold_Sample_Increase(index, data):
    
    # Use KFold to get more results
    k_10_fold_cv = KFold(n_splits = 10, random_state = 25, shuffle = True)
    print(f'We are using {k_10_fold_cv.get_n_splits(index)} fold cross validation')
    for i, (train_index, test_index) in enumerate(k_10_fold_cv.split(index_cv)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"  Test:  index={test_index}")
    # Concadenate all the datasets into one

    concat_train_data = []
    concat_test_data = []

    for i, (train_index, test_index) in enumerate(k_10_fold_cv.split(index_cv)):
        train_index = train_index.tolist()
        test_index = test_index.tolist()
        concat_train_data.append(train_index)
        concat_test_data.append(test_index)
        print(i)

    # Collapse the list of lists
    concat_final_data = concat_train_data + concat_test_data
    final_data = sum(concat_final_data, [])

    # Check that all distinct values of the original data are in the new dataset
    boolean_check = pd.Series(index_cv).drop_duplicates().tolist().sort() == pd.Series(final_data).drop_duplicates().tolist().sort()

    print(f"Concat with cross validation completed, the new data is {len(final_data)} records long and it is {str(boolean_check).lower()} that all original values are present")

    return final_data

# Ingest the testing results 
start_data_202 = pd.read_excel("C:/Users/andre/Downloads/DAEN_690.xlsx", sheet_name = 'Version202_ALL')

start_data_201 = pd.read_excel("C:/Users/andre/Downloads/DAEN_690.xlsx", sheet_name = 'Version201_ALL')

start_data_main = pd.read_excel("C:/Users/andre/Downloads/DAEN_690_INFERENCE_CAD.xlsx", sheet_name = None)

# If these files are not available, please download them from the following URLs and replace the reference
# If using the downloads folder, then the only requirement is to change the user name after downloading the files

url1 = "https://github.com/leezael/VisIQ/blob/main/src/DAEN_690.xlsx"

start_data_concat = pd.concat(start_data_main, ignore_index = True)

# Drop the first three columns

start_data_concat.drop( columns = start_data_concat.columns[1:3], inplace = True)

# Drop the first eight rows

start_data_concat.drop([0, 1, 2, 3, 4, 5, 6, 7, 8], inplace = True)

df_list = [start_data_202, start_data_201, start_data_concat]

start_data = pd.concat(df_list)

# Get the index that will be used to iterate through in K Fold Cross Validation
index_cv = start_data["IMAGE_NAME"]

# Call the custom function
try:
    final_data = K_Fold_Sample_Increase(index_cv, start_data)
except: 
    print("Something went wrong with the function, review the 10 Fold Cross Validation") 
else: 
     print("Function executed correctly")
# Start the K-Means approach
# Create random numbers for the columns of final_data
final_df = pd.DataFrame()
final_df["Index"] = final_data
# Join the original dataset values using merge
full_data = final_df.merge(start_data, left_on = "Index", right_index = True)
# Convert the model version and final dataset ID into categorical variables
full_data["MODEL_VERSION"] = full_data["MODEL_VERSION"].astype(str)
full_data["ACTUAL_DATASET_ID"] = full_data["ACTUAL_DATASET_ID"].astype(str)
full_data["PREDICTED_DATASET_ID"] = full_data["PREDICTED_DATASET_ID"].astype(str)

# Subset the data for K-Means
K_Means_data = full_data[["MODEL_NAME", "MATCH", "CONFIDENCE", "MODEL_VERSION", "ACTUAL_DATASET_ID"]]

full_data.rename(columns = {"IMAGE_NAME_y" : "IMAGE_NAME"}, inplace = True)

# Create the MATCH_NUMERIC column
full_data["MATCH_NUM"] = np.where(
    full_data["MATCH"] == 'YES', 1, 0
    )

# Create dumy columns for Model Name and Match
K_Means_data_dummies = pd.get_dummies(K_Means_data)
# Elbow method

inertias = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, random_state = i)
    kmeans.fit(K_Means_data_dummies)
    inertias.append(kmeans.inertia_)
    

plt.plot(range(1,15), inertias, marker='o')
plt.title('Elbow method graph')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Go with two clusters
def_kmeans = KMeans(n_clusters = 2, random_state = 235)

def_kmeans.fit(K_Means_data_dummies)

# Get the labels
clusters = def_kmeans.labels_

# Assign the clusters to the main dataset
full_data["Cluster"] = clusters

# Unique list of datasets
unique_datasets = full_data["ACTUAL_DATASET_ID"].unique()

# Start the ANOVA analysis using scipy
# Create the two datasets to use by using the confidence, and then see their size
first_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[0]]
second_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[1]]
third_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[2]]
fourth_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[3]]
fifth_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[4]]
sixth_ANOVA_df = full_data[full_data["ACTUAL_DATASET_ID"] == unique_datasets[5]]

first_ANOVA_pre = first_ANOVA_df["MATCH_NUM"] 
second_ANOVA_pre = second_ANOVA_df["MATCH_NUM"]
third_ANOVA_pre = third_ANOVA_df["MATCH_NUM"] 
fourth_ANOVA_pre = fourth_ANOVA_df["MATCH_NUM"] 
fifth_ANOVA_pre = fifth_ANOVA_df["MATCH_NUM"] 
sixth_ANOVA_pre = sixth_ANOVA_df["MATCH_NUM"]  

# Perform the ANOVA test
anova_prob = f_oneway(first_ANOVA_pre, second_ANOVA_pre, third_ANOVA_pre,
                      fourth_ANOVA_pre, fifth_ANOVA_pre, sixth_ANOVA_pre)

# Use seaborn to see a boxplot of the data
boxplot = sns.boxplot(x = 'MODEL_VERSION', y = 'CONFIDENCE', data = full_data, hue = 'ACTUAL_DATASET_ID')
sns.move_legend(boxplot, "upper left", bbox_to_anchor = (1, 1))
plt.plot()

# Now do the same with the clusters in the ANOVA and boxplot
first_ANOVA_df_c = full_data[full_data["Cluster"] == 0]
second_ANOVA_df_c = full_data[full_data["Cluster"] == 1]

first_ANOVA_pre_c = first_ANOVA_df_c["MATCH_NUM"] 
second_ANOVA_pre_c = second_ANOVA_df_c["MATCH_NUM"]

anova_prob_cluster = f_oneway(first_ANOVA_pre_c, second_ANOVA_pre_c)

# Boxplot by cluster
boxplot_c = sns.boxplot(x = 'Cluster', y = 'CONFIDENCE', data = full_data, hue = 'ACTUAL_DATASET_ID')
sns.move_legend(boxplot_c, "upper left", bbox_to_anchor = (1, 1))
plt.plot()

# Boxplot by model versions and actual dataset ID
boxplot_conf = sns.boxplot(x = 'MODEL_VERSION', y = 'CONFIDENCE', data = full_data, hue = 'ACTUAL_DATASET_ID')
plt.xticks(rotation=90)
sns.move_legend(boxplot_conf, "upper left", bbox_to_anchor = (1, 1))
plt.plot()

# Get the continuous variables from the full data
continuous_var = full_data.select_dtypes(include = 'number')
# Drop the Index
continuous_var.drop(columns = ['Index'], inplace = True)
# Correlation Matrix for continuous variables
corr_mat = continuous_var.corr()
# Mask the upper corner
mask = np.triu(np.ones_like(corr_mat, dtype=bool))
sns.heatmap(corr_mat, mask = mask)
plt.show()
# Q-Q plots for continuous variables
pplot(continuous_var, x = "CONFIDENCE", y = exponnorm, kind = 'qq')
plt.show()

pplot(continuous_var, x = "MATCH_NUM", y = exponnorm, kind = 'qq')
plt.show()

pplot(continuous_var, x = "Cluster", y = exponnorm, kind = 'qq')
plt.show()

# Linear Regression model on Confidence
# Get dummies from all variables

y = K_Means_data_dummies['CONFIDENCE']
x = K_Means_data_dummies.drop(['CONFIDENCE'], axis = 1)

Regressor = linear_model.LinearRegression()

# Train the model 
Regressor.fit(x, y)

# Create the regression table

regression_table = pd.DataFrame(list(x.columns)).copy()
regression_table.insert(len(regression_table.columns),"Coefficients", Regressor.coef_.transpose())
regression_table.rename(columns = {0 : "Features"}, inplace = True)

print(regression_table)

# Filter the table to the top 5 regressor features by using the absolute value
regression_table["Absolute Coefficients"] = regression_table["Coefficients"].abs()
regression_table.sort_values(by = ["Absolute Coefficients"], ascending = False)

regression_table.head(3)

# Print the score of the model
reg_pred = Regressor.predict(x)

print(Regressor.score(x, y))

# Print the MSE as well

print(mean_squared_error(y, reg_pred))

# Fit a decision tree regressor on Confidence

TreeReg = tree.DecisionTreeRegressor()
TreeReg.fit(x, y)
Tree_Reg_pred = TreeReg.predict(x)

# Print the accuracy of the regressor tree
print(TreeReg.score(x, y))
tree_importance_reg = TreeReg.feature_importances_

# Plot the tree importances sorted
tree_importances_reg = pd.Series(tree_importance_reg, index = x.columns)
tree_importances_reg.sort_values(ascending = False, inplace = True)
plt.title("Decision Tree Regressor Variable Importance")
tree_importances_reg.plot.bar()
plt.show()

# The score is a bit higher than the linear regression model, although still low to describe confidence besides the match correlation

Knn = neighbors.KNeighborsRegressor(n_neighbors = 10)

Knn.fit(x, y)

knn_pred = Knn.predict(x)

# Print KNN accuracy

print(mean_squared_error(y, knn_pred))

# Try with five neighbors instead, as this model is worse than the linear regression model

Knn_5 = neighbors.KNeighborsRegressor(n_neighbors = 5)

Knn_5.fit(x, y)

knn_pred_5 = Knn_5.predict(x)

# Print KNN accuracy

print(mean_squared_error(y, knn_pred_5))

# Try with 20 neighbors to finish the test of the KNN model

Knn_20 = neighbors.KNeighborsRegressor(n_neighbors = 20)

Knn_20.fit(x, y)

knn_pred_20 = Knn_20.predict(x)

# Print KNN accuracy

print(mean_squared_error(y, knn_pred_20))

# KNN is not capable of having a lower ME than regular regression.

# Fit a random forest

Forest_reg = ensemble.RandomForestRegressor()

Forest_reg.fit(x, y)

Forest_reg_pred = Forest_reg.predict(x)

# Print KNN accuracy

print(mean_squared_error(y, Forest_reg_pred))

# Logistic Regression with Cluster and Match being the target
Log_Regressor = linear_model.LogisticRegression(solver='lbfgs', max_iter=3000)
    
Log_Regressor.fit(K_Means_data_dummies, clusters)

Log_Regressor.coef_

# Print the accuracy of the Logistic Regression Model
print(Log_Regressor.score(K_Means_data_dummies, clusters))

# Print the Logistic Regression Table
log_table = pd.DataFrame(list(K_Means_data_dummies.columns)).copy()
log_table.insert(len(log_table.columns),"Coefficients", Log_Regressor.coef_.transpose())
log_table.rename(columns = {0 : "Features"}, inplace = True)
log_table["Absolute Coefficients"] = log_table["Coefficients"].abs()
log_table.sort_values(by = ["Absolute Coefficients"], ascending = False)

log_table.head(5)

# Train the model on Match instead of Cluster
Log_Regressor_Match = linear_model.LogisticRegression(solver='lbfgs', max_iter=3000)

# Drop the match variable
x_match = K_Means_data_dummies.drop(columns = ["MATCH_YES", "MATCH_NO"])
Log_Regressor_Match.fit(x_match, full_data["MATCH_NUM"])

Log_Regressor_Match.coef_

# Print the accuracy of the Logistic Regression Model
print(Log_Regressor_Match.score(x_match, full_data["MATCH_NUM"]))

# Print the Logistic Regression Table
log_table_match = pd.DataFrame(list(x_match.columns)).copy()
log_table_match.insert(len(log_table_match.columns),"Coefficients", Log_Regressor_Match.coef_.transpose())
log_table_match.rename(columns = {0 : "Features"}, inplace = True)
log_table_match["Absolute Coefficients"] = log_table_match["Coefficients"].abs()
log_table_match.sort_values(by = ["Absolute Coefficients"], ascending = False)

log_table_match.head(5)

# Train a classification tree with Match

TreeClassifier = tree.DecisionTreeClassifier()
TreeClassifier.fit(x_match, full_data["MATCH_NUM"])

# Print the accuracy of the Classification Tree
print(TreeClassifier.score(x_match, full_data["MATCH_NUM"]))
tree_importance = TreeClassifier.feature_importances_

# Plot the tree importances sorted
tree_importances = pd.Series(tree_importance, index = x_match.columns)
tree_importances.sort_values(ascending = False, inplace = True)
plt.title("Decision Tree Variable Importance")
tree_importances.plot.bar()
plt.show()

# Train a random forest with Match
RD_CLASS = ensemble.RandomForestClassifier()
RD_CLASS.fit(x_match, full_data["MATCH_NUM"])

# Print the accuracy of the Random Forest Classifier
print(RD_CLASS.score(x_match, full_data["MATCH_NUM"]))
forest_importance = RD_CLASS.feature_importances_

# Plot the forest importances sorted
forest_importances = pd.Series(forest_importance, index = x_match.columns)
forest_importances.sort_values(ascending = False, inplace = True)
plt.title("Random Forest Variable Importance")
forest_importances.plot.bar()
plt.show()