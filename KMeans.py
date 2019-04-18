import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

originalData = pd.read_csv("FinalMentorResponses.csv")
data=originalData.copy()

print(data)
data.iloc[0:19:,:]['Email Address']='abc'
print(data.iloc[0:19:,:]['Email Address'])

activities=["Art","Board Games","Hiking/Outdoors","Greek Life","Music","Partying","Reading","Sports","Video Games","Watching TV/Movies"]
A1='1st fav activity';
A2='2nd fav activity';
A3='3rd fav activity';
A4='4th fav activity';
A5='5th fav activity';
data.loc[data[A1] == 'Art', A1] = 1
data.loc[data[A1] == 'Board Games', A1] = 2
data.loc[data[A1] == 'Hiking/Outdoors', A1] = 3
data.loc[data[A1] == 'Greek Life', A1] = 4
data.loc[data[A1] == 'Music', A1] = 5
data.loc[data[A1] == 'Partying', A1] = 6
data.loc[data[A1] == 'Reading', A1] = 7
data.loc[data[A1] == 'Sports', A1] = 8
data.loc[data[A1] == 'Video Games', A1] = 9
data.loc[data[A1] == 'Watching TV/Movies', A1] = 10

data.loc[data[A2] == 'Art', A2] = 1
data.loc[data[A2] == 'Board Games', A2] = 2
data.loc[data[A2] == 'Hiking/Outdoors', A2] = 3
data.loc[data[A2] == 'Greek Life', A2] = 4
data.loc[data[A2] == 'Music', A2] = 5
data.loc[data[A2] == 'Partying', A2] = 6
data.loc[data[A2] == 'Reading', A2] = 7
data.loc[data[A2] == 'Sports', A2] = 8
data.loc[data[A2] == 'Video Games', A2] = 9
data.loc[data[A2] == 'Watching TV/Movies', A2] = 10

data.loc[data[A3] == 'Art', A3] = 1
data.loc[data[A3] == 'Board Games', A3] = 2
data.loc[data[A3] == 'Hiking/Outdoors', A3] = 3
data.loc[data[A3] == 'Greek Life', A3] = 4
data.loc[data[A3] == 'Music', A3] = 5
data.loc[data[A3] == 'Partying', A3] = 6
data.loc[data[A3] == 'Reading', A3] = 7
data.loc[data[A3] == 'Sports', A3] = 8
data.loc[data[A3] == 'Video Games', A3] = 9
data.loc[data[A3] == 'Watching TV/Movies', A3] = 10

data.loc[data[A4] == 'Art', A4] = 1
data.loc[data[A4] == 'Board Games', A4] = 2
data.loc[data[A4] == 'Hiking/Outdoors', A4] = 3
data.loc[data[A4] == 'Greek Life', A4] = 4
data.loc[data[A4] == 'Music', A4] = 5
data.loc[data[A4] == 'Partying', A4] = 6
data.loc[data[A4] == 'Reading', A4] = 7
data.loc[data[A4] == 'Sports', A4] = 8
data.loc[data[A4] == 'Video Games', A4] = 9
data.loc[data[A4] == 'Watching TV/Movies', A4] = 10

data.loc[data[A5] == 'Art', A5] = 1
data.loc[data[A5] == 'Board Games', A5] = 2
data.loc[data[A5] == 'Hiking/Outdoors', A5] = 3
data.loc[data[A5] == 'Greek Life', A5] = 4
data.loc[data[A5] == 'Music', A5] = 5
data.loc[data[A5] == 'Partying', A5] = 6
data.loc[data[A5] == 'Reading', A5] = 7
data.loc[data[A5] == 'Sports', A5] = 8
data.loc[data[A5] == 'Video Games', A5] = 9
data.loc[data[A5] == 'Watching TV/Movies', A5] = 10

IdF='What most describes an ideal Friday for you'
data.loc[data[IdF] == 'Party', IdF] = 1
data.loc[data[IdF] == 'Catch up on sleep', IdF] = 2
data.loc[data[IdF] == 'Get ahead in class/Finish up homework', IdF] = 3
data.loc[data[IdF] == 'Hang out with friends', IdF] = 4

TR='Which of the following traits to you value most in a mentor'
data.loc[data[TR] == 'Trustworthiness', TR] = 1
data.loc[data[TR] == 'Wisdom', TR] = 2
data.loc[data[TR] == 'Kindness', TR] = 3
data.loc[data[TR] == 'Sense of humor', TR] = 4
data.loc[data[TR] == 'Confidence', TR] = 5

BG='Would you rather hang out in a big group or one on one with someone?'
data.loc[data[BG] == 'Big group', BG] = 1
data.loc[data[BG] == 'One on one', BG] = 2

MJ='What is your major?'
data.loc[data[MJ] == 'Computer Science', MJ] = 1
data.loc[data[MJ] == 'Computer Science and Engineering', MJ] = 1
data.loc[data[MJ] == 'Computer Engineering', MJ] = 1
data.loc[data[MJ] == 'Electrical Engineering', MJ] = 1
data.loc[data[MJ] == 'Aerospace Engineering', MJ] = 2
data.loc[data[MJ] == 'Mechanical Engineering', MJ] = 2
data.loc[data[MJ] == 'Material Science and Engineering', MJ] = 3
data.loc[data[MJ] == 'Chemical Engineering', MJ] = 3
data.loc[data[MJ] == 'Civil & Environmental Engineering', MJ] = 4
data.loc[data[MJ] == 'Bioengineering', MJ] = 4
data.loc[data[MJ] == 'Cognitive Science', MJ] = 5

data[A1]=data[A1]*10
data[A2]=data[A2]*8
data[A3]=data[A3]*6
data[A4]=data[A4]*4
data[A5]=data[A5]*2
data[IdF]=data[IdF]*5
data[TR]=data[TR]*5
data[BG]=data[BG]*7
data[MJ]=data[MJ]*20

newData=data[[A1,A2,A3,A4,A5,IdF,TR,BG,MJ]].copy()
kmeans = KMeans(n_clusters=3,n_init=200)
# Fitting the input data
kmeans = kmeans.fit(newData)
# Getting the cluster labels
labels = kmeans.predict(newData)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)
print(labels)


for i in range(0,19):
    if(labels[i]==0):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")

for i in range(0,19):
    if(labels[i]==1):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")

for i in range(0,19):
    if(labels[i]==2):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")

for i in range(0,19):
    if(labels[i]==3):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")

for i in range(0,19):
    if(labels[i]==4):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")

for i in range(0,19):
    if(labels[i]==5):
        print(originalData.loc[i,:])

print("--------------------------------------------------------")








print(originalData.iloc[0:19:,:][A1])
print(originalData.iloc[0:19:,:][A2])
print(originalData.iloc[0:19:,:][A3])
print(originalData.iloc[0:19:,:][A4])
print(originalData.iloc[0:19:,:][A5])
print(data)
