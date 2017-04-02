import pandas as pd
import numpy as np


def Aggregate(inFileName, outFileName):
	df=pd.read_csv(inFileName,header=0)
	copy = df.copy()

	del df['srch_destination_id']
	del df['srch_destination_name']
	del	df['srch_destination_type_id']
	del	df['srch_destination_latitude']
	del df['srch_destination_longitude']

	#Remove log bases so summation makes sense
	exp = lambda x: 10**x
	df = df.apply(exp) 

	categories = set([name.split('_')[1] for name in df.columns])


	for name in df.columns:
		if name.split('_')[1] not in df:
			df[name.split('_')[1]] = df[name] 
		else:
			df[name.split('_')[1]]+= df[name]


	for name in df.columns:
		if name not in categories:
			del df[name]

	df.insert(0, 'srch_destination_longitude', copy['srch_destination_longitude'])
	df.insert(0, 'srch_destination_latitude', copy['srch_destination_latitude'])
	df.insert(0, 'srch_destination_type_id', copy['srch_destination_type_id'])		
	df.insert(0, 'srch_destination_name', copy['srch_destination_name'])
	df.insert(0, 'srch_destination_id', copy['srch_destination_id'])


	df.to_csv(outFileName)


if __name__ == '__main__':
	inFileName = 'top100Dest.csv'
	outFileName = 'Attributes.csv'
	Aggregate(inFileName, outFileName)




