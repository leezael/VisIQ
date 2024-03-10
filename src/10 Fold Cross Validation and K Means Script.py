# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 19:30:18 2024


"""

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


start_data = pd.read_excel("C:/Users/andre/Downloads/DAEN_690.xlsx", sheet_name = 'Version202_ALL')
# Ingest the testing results 

index_cv = start_data["IMAGE_NAME"]

# Check the dataframe


# Use KFold to get more results

k_10_fold_cv = KFold(n_splits = 10, random_state = 25, shuffle = True)

print(f'We are using {k_10_fold_cv.get_n_splits(index_cv)} fold cross validation')

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

print(f"Concat with cross validation completed, the new data is {len(final_data)} records long and it is {boolean_check} that all original values are present")

# Start the K-Means approach
# Create random numbers for the columns of final_data
final_df = pd.DataFrame()
final_df["Index"] = final_data
# Join the original dataset values using merge
full_data = final_df.merge(start_data, left_on = "Index", right_index = True)

# Subset the data for K-Means
K_Means_data = full_data[["MODEL_NAME", "MATCH"]]
K_Means_data.rename(columns = {"IMAGE_NAME_y" : "IMAGE_NAME"}, inplace = True)

# Create dumy columns for Model Name and Match
K_Means_data_dummies = pd.get_dummies(K_Means_data)
# Elbow method

inertias = []

for i in range(1, 15):
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(final_df[['attr1', 'attr2']])
    inertias.append(kmeans.inertia_)
    

plt.plot(range(1,15), inertias, marker='o')
plt.title('Elbow method graph')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Start the ANOVA analysis using scipy