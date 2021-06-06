import sys, os
from bayes import Bayes_Classifier

testDir = "reviews/movie_and_product_reviews/"
trainDir = "reviews/movie_and_product_reviews/"

bc = Bayes_Classifier(trainDir)

testDir = sys.argv[1]

for fFileObj in os.walk(testDir):
	iFileList = fFileObj[2]
	break

testing_data = iFileList

# model_information = bc.load("model_information")
# testing_data = model_information["testing_files"]

'''	For binary classification:
'''
# tp = 0
# tn = 0
# fp = 0
# fn = 0

# for data in testing_data:
# 	rating = data.split('-')[1] 
# 	text = bc.loadFile(trainDir + data)
# 	result =  bc.classify(text)

# 	if result == "positive" and rating == '5':
# 		tp += 1
# 	elif result == "positive" and rating == '1':
# 		fp += 1
# 	elif result == "negative" and rating == '5':
# 		fn += 1
# 	elif result == "negative" and rating == '1':
# 		tn += 1

# 	precision = tp / (tp + fp)
# 	recall = tp / (tp + fn)
# 	fmeasure = 2 * ((precision * recall) / (precision + recall))
# 	accuracy = (tp + tn) / (tp + tn + fp + fn)

# 	print("Precision :", precision)
# 	print("Recall :", recall)
# 	print("F-measure :", fmeasure)
# 	print("Accuracy", accuracy)

''' Since we have multiple classes, we shall use macroaverging to determine the metrics of each class.
	We create a confusion matrix as follows:

				positive neutral negative
	positive	[	0		0		0	]
	neutral		[	0		0		0	]
	negative	[	0		0		0	]

'''
confusion_matrix = [[0,0,0],[0,0,0],[0,0,0]]

'''	Treating '5' as positive class, '1' as negative class, ['2','3','4'] as neutral class.
'''
# for data in testing_data:
# 	rating = data.split('-')[1] 
# 	text = bc.loadFile(trainDir + data)
# 	result =  bc.classify(text)

# 	if result == "positive" and rating == '5':
# 		confusion_matrix[0][0] += 1
# 	elif result == "positive" and rating in ['2','3','4']:
# 		confusion_matrix[0][1] += 1
# 	elif result == "positive" and rating == '1':
# 		confusion_matrix[0][2] += 1
# 	elif result == "neutral" and rating == '5':
# 		confusion_matrix[1][0] += 1
# 	elif result == "neutral" and rating in ['2','3','4']:
# 		confusion_matrix[1][1] += 1
# 	elif result == "neutral" and rating == '1' :
# 		confusion_matrix[1][2] += 1
# 	elif result == "negative" and rating == '5':
# 		confusion_matrix[2][0] += 1
# 	elif result == "negative" and rating in ['2','3','4']:
# 		confusion_matrix[2][1] += 1
# 	elif result == "negative" and rating == '1':
# 		confusion_matrix[2][2] += 1

'''	Treating ['4','5'] as positive class, ['1','2'] as negative class, '3' as neutral class.
'''

for data in testing_data:
	rating = data.split('-')[1] 
	text = bc.loadFile(trainDir + data)
	result =  bc.classify(text)

	if result == "positive" and rating in  ['4','5']:
		confusion_matrix[0][0] += 1
	elif result == "positive" and rating == '3':
		confusion_matrix[0][1] += 1
	elif result == "positive" and rating in ['1','2']:
		confusion_matrix[0][2] += 1
	elif result == "neutral" and rating in  ['4','5']:
		confusion_matrix[1][0] += 1
	elif result == "neutral" and rating == '3':
		confusion_matrix[1][1] += 1
	elif result == "neutral" and rating in ['1','2'] :
		confusion_matrix[1][2] += 1
	elif result == "negative" and rating in  ['4','5']:
		confusion_matrix[2][0] += 1
	elif result == "negative" and rating  == '3':
		confusion_matrix[2][1] += 1
	elif result == "negative" and rating in ['1','2']:
		confusion_matrix[2][2] += 1

precision = []
i = 0
for row in confusion_matrix:
	precision.append(row[i] / sum(row))
	i += 1

recall = []
i = 0
for row in confusion_matrix:
	recall.append(row[i] / sum([row[i] for row in confusion_matrix]))
	i += 1

a = sum([confusion_matrix[i][i] for i in range(len(confusion_matrix))]) / len(testing_data)
p = sum(precision) / len(precision)
r = sum(recall) / len(recall)
f = (p * r) / (p + r)

print("Precision :", p)
print("Recall :", r)
print("F-measure :", f)
print("Accuracy", a)

print('%d test reviews.' % len(testing_data)) 

# results = {"negative":0, "neutral":0, "positive":0}

# print("\nFile Classifications:")
# for filename in iFileList:
# 	fileText = bc.loadFile(testDir + filename)
# 	result = bc.classify(fileText)
# 	print("%s: %s" % (filename, result))
# 	results[result] += 1

# print("\nResults Summary:")
# for r in results:
# 	print("%s: %d" % (r, results[r]))