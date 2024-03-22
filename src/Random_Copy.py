import os
import random
import shutil
no_of_files = 10


source = '/Users/evp/Documents/GMU/DAEN690/Datasets/Brain_Classification/Files/Training/notumor/'
dest = '/Users/evp/Documents/GMU/DAEN690/Intellidetect_git/VisIQ/datasets/Brain_Classification/N500/Set3'
files = os.listdir(source)
if not os.path.exists(dest):
    os.makedirs(dest)

for file_name in random.sample(files, no_of_files):
    shutil.copyfile(os.path.join(source, file_name), os.path.join(dest, file_name))