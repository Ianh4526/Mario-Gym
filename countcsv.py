import csv
import numpy as np

with open("MDS.csv") as f:
    reader = csv.reader(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    print 'loading...'
    np.set_printoptions(threshold='nan')
    for row in reader:
        print(row)
        print (len(row))
        
