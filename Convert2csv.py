import csv
import numpy as np

inputFile = 'data.txt'
outputFile = 'NewData.csv'

with open(inputFile) as fin, open(outputFile, 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split('	'))
    print "Converted" + inputFile+ " to " + outputFile
