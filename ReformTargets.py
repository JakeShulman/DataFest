import csv
import pandas as pd
import numpy as np
import pickle

inFile = 'Target.csv'
BookingData = 'BookingCumSum.csv'
LookingData = 'LookingCumSum.csv'
outFile = 'ReformedTarget.csv'

def cityIDs(inFile):
	cityKeys = dict()
	df=pd.read_csv(inFile,header=0)
	for ID in range(len(df['srch_destination_id'].values)):
		cityKeys[df['srch_destination_id'].values[ID]] = df['srch_destination_name'].values[ID]
	pickle.dump(cityKeys, open('keys.p', 'wb'))
	return cityKeys

cityIDs('top100Dest.csv')


df=pd.read_csv(inFile,header=0)
#Bookings
BookingDF = pd.read_csv(BookingData, header = 0)
BookingDF = BookingDF.set_index('Dates')
bookheaders = BookingDF.columns
bookingArray = BookingDF.values
outbookArray = []
for day in range(len(bookingArray)):
	for column in range(len(bookheaders)):
		outbookArray.append(bookingArray[day][column])

LookingDF = pd.read_csv(LookingData, header = 0)
LookingDF = LookingDF.set_index('Dates')
lookingHeader = LookingDF.columns
lookingArray = LookingDF.values
outlookArray = []
for day in range(len(lookingArray)):
	for column in range(len(lookingHeader)):
		outlookArray.append(lookingArray[day][column])



array = df.values
headers = df.columns[1:]

cityIDs = pickle.load(open('keys.p','rb'))

outArray = []
count = 0
for day in range(len(array)):
	for column in range(len(headers)):
		outArray.append([array[day][0], cityIDs[int(headers[column])], headers[column], array[day][column+1], outbookArray[count], outlookArray[count]])
		count+=1

outArray = np.array(outArray)



df = pd.DataFrame(outArray, index = [x for x in range(len(outArray))])

df.to_csv(outFile)
