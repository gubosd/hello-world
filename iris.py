# iris.py

import numpy as np

f=open('iris.csv')

line=f.readline()
header=line.strip().split(',')
header=[i.strip('"') for i in header]

data=[]
label=[]

for line in f:
	l=line.strip().split(',')
	dl=[float(i) for i in l[:4]]
	
	data.append(dl)
	label.append(l[4].strip('"'))
	
X=np.array(data); print(X.shape)
y=np.array(label); print(y.shape)

f.close()

# save to 'iris2.csv'
f=open('iris2.csv','w')
f.write(','.join(header) + '\n')
for i,j in zip(X,y):
	f.write('%f,%f,%f,%f,%s\n' % (*i,j))
f.close()