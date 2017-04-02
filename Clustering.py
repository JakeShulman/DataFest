from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances_argmin_min
import matplotlib.pyplot as plt
from InterestAggregate import Aggregate
import numpy as np
import pandas as pd


def extract(inFile, intermediateFile):
	#Use interest aggreation from InterestAggregate.py, sums probabillities over activity type
	Aggregate(inFile, intermediateFile)

	#Format data for clustering
	NewDest=pd.read_csv(intermediateFile)
	#Remove numbered rows (artifact of data frames)
	NewDest.drop(NewDest.columns[0], axis=1, inplace=True)

	copy=NewDest.copy()
	del NewDest['srch_destination_id']
	del NewDest['srch_destination_name']
	del NewDest['srch_destination_type_id']
	del NewDest['srch_destination_latitude']
	del NewDest['srch_destination_longitude']

	NewDest=NewDest.dropna()
	return NewDest, copy


def cluster(Data):
	copy = Data[1]
	Data = Data[0]
	cluster= AffinityPropagation().fit(Data)
	# closest, _ = pairwise_distances_argmin_min(cluster.cluster_centers_, Data)
	Labels = set(cluster.labels_ )
	se = pd.Series(cluster.labels_)
	copy['Labels'] = se.values
	copy2=copy.as_matrix()
	Data['Labels']=se.values

	Data.insert(0, 'srch_destination_longitude', copy['srch_destination_longitude'])
	Data.insert(0, 'srch_destination_latitude', copy['srch_destination_latitude'])
	Data.insert(0, 'srch_destination_type_id', copy['srch_destination_type_id'])		
	Data.insert(0, 'srch_destination_name', copy['srch_destination_name'])
	Data.insert(0, 'srch_destination_id', copy['srch_destination_id'])
	# for x in closest:
	# 	print copy2[x][1:5],copy2[x][-1]

	Data.to_csv('TopDestLabeled.csv')
	return Data, Labels

def parseLabels(LabeledData, Labels):
	fnames = []
	for label in Labels:
		partialData = LabeledData.loc[LabeledData['Labels'] == label]
		fnames.append('Cluster'+str(label)+'.csv')
		partialData.to_csv('Cluster'+str(label)+'.csv')
		print "Cluster ", label, " Parsed into CSV"
	return fnames

def main(inFile, intermediateFile):
	Data = extract(inFile, intermediateFile)
	LabeledData = cluster(Data)
	fnames = parseLabels(LabeledData[0], LabeledData[1])
	return fnames

if __name__ == '__main__':
	inFile = 'top100Dest.csv'
	intermediateFile = 'Attributes.csv'
	main(inFile, intermediateFile)	















