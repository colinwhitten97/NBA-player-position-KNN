import pandas as pd
import numpy as np
from scipy import spatial
import operator

# load csv data
mydf = pd.read_csv('2007train2.csv')
testdf = pd.read_csv('2004test.csv')

# split csv data into two dataframes, the class and attributes
df_entities = mydf.iloc[:,[2]]
#df_attributes = mydf.iloc[:, [0,1,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
df_attributes = mydf.iloc[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]
# merge entities and attributes into one dataframe
df_merged = pd.concat([df_entities, df_attributes], axis = 1 , sort = False)

# convert dataframe to array
df_array = df_merged.to_numpy()


playerdict = {}
entry_num = 0

for d in df_array:
  playerID = entry_num
  position = int(d[0])
  attributes = d[1:]
  attributes = map(int, attributes)
  playerdict[playerID] = (position, np.array(list(attributes)))
  entry_num += 1

# Compute Neighbors:

def getNeighbors(playerID, K):
  distances = []
  for player in playerdict:
    if (player != playerID):
      dist = ComputeDistance(playerdict[playerID], playerdict[player])
      distances.append((player, dist))
  distances.sort(key = operator.itemgetter(1))

  neighbors = []
  for i in range(K):
    neighbors.append((distances[i][0]))
  return neighbors


def ComputeDistance(a, b):
    dataA = a[1]
    dataB = b[1]

    AttributeDistance = spatial.distance.cosine(dataA, dataB)

    return AttributeDistance


K = 5


count = 0
for i in range(len(df_array)):
  neighbors = getNeighbors(i,K)
  poss = []
  for n in neighbors:
    test = df_array[n][0]
    poss.append(test)
  guess = int(max(set(poss), key = poss.count))
  if guess == int(df_array[i][0]):
    count += 1
acc = count/len(df_array)
print(acc)
  
  
# END OF FIRST PART!

def getNeighbors2(playerID, K):
  distances = []
  playerdict[playerID] = testdict[playerID]
  for player in playerdict:
    if (player != playerID):
      dist = ComputeDistance(playerdict[playerID], playerdict[player])
      distances.append((player, dist))
  distances.sort(key = operator.itemgetter(1))

  neighbors = []
  for i in range(K):
    neighbors.append((distances[i][0]))
  del playerdict[playerID]
  return neighbors

# split csv data into two dataframes, the class and attributes
df_entities = testdf.iloc[:,[2]]
df_attributes = testdf.iloc[:, [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]]

# merge entities and attributes into one dataframe
df_merged = pd.concat([df_entities, df_attributes], axis = 1 , sort = False)

# convert dataframe to array
df_array0 = df_merged.to_numpy()

testdict = {}
entry_num = len(df_array) + 1
nn = entry_num
for d in df_array0:
  playerID = entry_num
  position = int(d[0])
  attributes = d[1:]
  attributes = map(int, attributes)
  testdict[playerID] = (position, np.array(list(attributes)))
  entry_num += 1

testlist = []
validlist = []
for nums in range(len(df_array0)):
  validlist.append(df_array0[nums][0])
  #make this cleaner- get 0th column of matrix

for t in range(nn,nn+len(df_array0)):
  neighborz = getNeighbors2(t, K)
  poss = []
  for p in neighborz:
    test = df_array[p][0]
    poss.append(test)
  guess = int(max(set(poss), key = poss.count))
  testlist.append(guess)
  

right = 0
for i in range(len(testlist)):
  if testlist[i] == validlist[i]:
    right+= 1

acc2 = right/ len(df_array0)
print(acc2)


