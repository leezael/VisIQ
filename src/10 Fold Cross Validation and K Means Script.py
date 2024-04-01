# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:30:18 2024
Main Python Script for 10 Fold Cross Validation, K Means and ANOVA analysis

"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
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

# Boxplot
boxplot_c = sns.boxplot(x = 'Cluster', y = 'CONFIDENCE', data = full_data, hue = 'ACTUAL_DATASET_ID')
sns.move_legend(boxplot_c, "upper left", bbox_to_anchor = (1, 1))
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