# -*- coding: utf-8 -*-
"""

"""
import glob
import csv

folderToRead = 'C:/Users/mestrovicd/Desktop/Wissen/Deep_Learning/TF_2_Notebooks_and_Data/Project/Images/_trimed/_1/'

filename = []

with open('C:/Users/mestrovicd/Desktop/Wissen/Deep_Learning/TF_2_Notebooks_and_Data/Project/Images/features.csv','w',newline='\n') as file:
    writer = csv.writer(file,delimiter=';')

    for filename in glob.glob(folderToRead+'*.jpg'):
        jpg_name = filename.replace('C:/Users/mestrovicd/Desktop/Wissen/Deep_Learning/TF_2_Notebooks_and_Data/Project/Images/_trimed/_1', '')
        write_into_file = jpg_name[1:len(jpg_name)]
        print(write_into_file)
        writer.writerow([write_into_file])

