# accuracy is 0.78
# output is in predictions.txt

# import all the libraries needed
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# open the train and test files
trainfile = open('train.data', 'r')
testfile = open('test.data','r')

# use readlines to read all the lines of the files to a list
traindata = trainfile.readlines()
testdata = testfile.readlines()

# use count vectorizer to tokenize and build a vocabulary of known words
vectorizer = CountVectorizer(analyzer='word', lowercase=True, stop_words='english')

# fit and transform the data for better use
triandatatransformed = vectorizer.fit_transform(traindata)
testdatatransformed = vectorizer.transform(testdata)

# using sklearn library compute the cosine similarity between two values
# we will use this to measure how similar the the reviews are regardless of their size
cosval = cosine_similarity(testdatatransformed,triandatatransformed)

# knn implementation

# open the document where all the predictions will go
knnoutput = open('predictions.txt','w')

# loop through every row value of the cosine similarity matrix
for x in cosval:
    k = 77 # 19 # 37
    # create a partitioned copy of array
    # elements rearranged according to the given k
    newarr = np.argpartition(-x, k)
    temparr = newarr[:k]

    # loop through the train data and record the sign of the neighbor
    # ctr to keep track
    sentiment = 0
    for y in temparr:
        if traindata[y].strip()[0] == '+':
            sentiment += 1
        elif traindata[y].strip()[0] == '-':
            sentiment -= 1

    # check the sentiment and write into the file
    if sentiment > 0:
        knnoutput.write('+1\n')
    else:
        knnoutput.write('-1\n')

print('end of predictions')
