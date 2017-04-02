from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
import Clustering
import csv
import numpy as np
import pandas as pd
from random import shuffle
import Clustering
import pickle
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore', category = DeprecationWarning)



cityKeys = dict()
def cityIDs(inFile):
	global cityKeys
	df=pd.read_csv(inFile,header=0)
	for ID in range(len(df['srch_destination_id'].values)):
		cityKeys[df['srch_destination_id'].values[ID]] = [df['srch_destination_name'].values[ID], df['Labels'].values[ID]]
	return cityKeys

def clusterData(inFile, masterFile):

	"""Creates new clusterFiles for each cluster the masterCSV jake passes which contains the 
	Date: city : inputs... : target number of rooms
	For each city in a given cluster file, find all of its occurences in MasterFile and 
	append them to the same new file. This file will be used for the Neural networks.
	"""

	intermediateFile = "Attributes.csv"
	clusterFiles = Clustering.main(inFile, intermediateFile)
	newclusterFiles = []
	for clusterFile in clusterFiles:
		cityIDs(clusterFile)

	df = pd.read_csv(masterFile, header=0)
	clusterIDs = []
	for city in range(len(df['City_ID'].values)):
		clusterIDs.append(cityKeys[df['City_ID'].values[city]][1])

	df['City_Labels'] = pd.Series(clusterIDs)
	df.to_csv(str('test')+masterFile)

	newFiles = []
	for cluster in set(df['City_Labels'].values):
		fname = "Training"+clusterFiles[cluster]
		partialData = df.loc[df['City_Labels'] == cluster]
		partialData.to_csv(fname)
		newFiles.append(fname)

	return newFiles

def extract(fileName, Ptrain, forecastRange):
	df=pd.read_csv(fileName,header=0)

	#Get the start and end row of the forecasting period
	forecastData = df.copy()
	forecastData = df.ix[forecastRange[0]:,:]
	forecastData.drop(forecastData.columns[0], axis=1, inplace=True)


	inputsArray = df.copy()
	inputsArray = inputsArray.head(n=forecastRange[0])
	inputsArray.drop(inputsArray.columns[0], axis=1, inplace=True)


	print "Cluster Size: ", len(inputsArray)/365.0
	print "Number of Cluster Samples: ", len(inputsArray)
	#Convert dataframes into input and target arrays FORECASTING
	copyF = forecastData.copy()
	del copyF['Rooms']
	inputsArrayForecast = copyF.values
	targetArrayForecast = forecastData['Rooms'].values

	copyF = inputsArray.copy()
	del copyF['Rooms']
	inputsArrayT = copyF.values
	targetArrayT = inputsArray['Rooms'].values

	#Randomize training & testing
	numRows = len(inputsArrayT)
	randRows = [x for x in range(0, numRows)]
	shuffle(randRows)
	trainRows = randRows[:int(numRows*Ptrain)]
	testRows = randRows[int(numRows*Ptrain):]

	#Seperate into training & testing
	inputsArrayTrain = inputsArrayT[trainRows]
	targetArrayTrain = targetArrayT[trainRows]

	inputsArrayTest = inputsArrayT[testRows]
	targetArrayTest = targetArrayT[testRows]

	# headers = df.columns
	# for x in inputsArrayTrain:
	# 	for i in range(len(headers)):
	# 		headers[i]


	return {
			'Xtrain': 	inputsArrayTrain,
			'Ytrain': 	targetArrayTrain, 
			'Xtest':  	inputsArrayTest,
			'Ytest':	targetArrayTest,
			'Xfuture': 	inputsArrayForecast,
			'Yfuture': 	targetArrayForecast
			}


def train(Data):
	X = [x[-4:-1] for x in Data['Xtrain']]
	Y = Data['Ytrain']
	numInputs = len(X[0])
	# clf = MLPRegressor(solver='lbfgs', alpha=1e-5,
	#                     hidden_layer_sizes=(numInputs*2, numInputs, 3), random_state=1)
	clf = SVR(kernel='rbf', C=1e3, gamma=1)

	clf.fit(X, Y)   
	print "SUCCESSFUL FIT"
	return clf

def test(Data, model):
	X = [x[-4:-1] for x in Data['Xtest']]
	Y = Data['Ytest']

	MSE = 0.0
	predictionDict = dict()

	for row in range(len(X)):
		date = Data['Xtest'][row][0]
		prediction = model.predict(X[row])
		actual = Y[row]
		predictionDict[date] = {'p': prediction, 'a': actual}
		MSE+= (actual - prediction)**2

	MSE/=len(X)
	print "MEAN SQUARED ERROR ON TESTING: ", MSE

	return predictionDict

def forecast(Data, model):
	X = [x[-4:-1] for x in Data['Xfuture']]
	Y = Data['Yfuture']

	MSE = 0.0
	predictionList = []
	for row in range(len(X)):
		date = Data['Xfuture'][row][0]
		city = Data['Xfuture'][row][1]
		prediction = model.predict(X[row])
		actual = Y[row]
		predictionList.append([date, city, prediction[0]])
		MSE+= (actual - prediction)**2

	return predictionList


def finalCSV(predictions):
	with open("Predictions.csv", 'wb') as f:
		for cluster in predictions:
			for row in cluster:
			    wr = csv.writer(f, dialect='excel')
			    wr.writerow(row)


def main(inFile, masterFile, Ptrain, forecastRange):
	files = clusterData(inFile, masterFile)
	predictions = []
	for clusterFile in files:
		#Extract the data from the preprocessed CSV and partition it into train, test, forecast
		Data = extract(clusterFile, Ptrain, forecastRange)
		#Train the neural network and create a model
		Model = train(Data)
		#Test how good the network is and get MSE
		testDict = test(Data, Model)
		#Forcast future Room booking demands
		predictions.append(forecast(Data, Model))
	#Export predictions from forecasting into a csv for visualization
	finalCSV(predictions)

if __name__ == '__main__':

	inFile= 'top100Dest.csv'
	masterFile = 'ReformedTargetNorm.csv'

	Ptrain = .80
	forecastRange = ['12/15/15', '1/4/16']
	forecastRange = [3500, 4014]

	main(inFile, masterFile, Ptrain, forecastRange)









