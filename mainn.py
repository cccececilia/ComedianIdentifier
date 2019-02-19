import nltk
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import random
from nltk.corpus import brown
import math
from operator import itemgetter

data = []
stop_words = set(stopwords.words('english')) #All meaningless words such as 'to', 'at'...

#basically split a sentence into words
def tokenize_raw_data(raw_data):
    print('tokenizing data')
    tokenized = []
    for row in raw_data:    #for each row in csvfile
        #only need to tokenize the joke column
        tokenized.append([row[0], row[1], row[2], word_tokenize(row[3])])
    return tokenized

#the program starts here
with open('textfiles/Final CSV  - Sheet1.csv') as file:
        print('\n\n')
        reader = csv.reader(file, delimiter=',')
        line = 0
        print('Reading data from csv file')
        for row in reader:
            if line != 0:
                data.append(row)
            line += 1
print('Reading complete')
tokenized_data = tokenize_raw_data(data)
print('Training...')
total = 0
feature = set() #used set so that there will be no duplicate data
for counter in range(500):  #train and test 100 times
    #print(counter)
    random.shuffle(tokenized_data) #shuffle the data so that we get a more accurate result
    feature_sets = [] #actual feature set
    for i in tokenized_data:
#        #i[3] is the actual joke, i[0] is name of comedian
#        #data cleaning and creating feature dictionary, True means we want this word to be used
#        word_feature = dict([(word.lower(), True) for word in i[3]
#        if word not in punctuation
#        and word.lower() not in ['trump', 'obama',  'laughter', 'cheering', 'applause',]
#        and word != '``'
#        and word != "''"
#        and word != '""'
#        and word.lower() not in stop_words])
#        feature_sets.append((word_feature, i[0])) #feature_set element: (dict of {word, True}, comedian_name)
#        #for word in feature_set
        word_occur = {}
        for word in i[3]:
            if word not in punctuation and word.lower() not in ['trump', 'obama',  'laughter', 'cheering', 'applause',] and word != '``' and word != "''" and word != '""' and word.lower() not in stop_words:
    #       validate
                if word not in word_occur:
                    word_occur.update(dict(word = 1))
                else:
                    word_occur[word] = word_occur[word] + 1
    #       if validated, check if in word occur
    #       if not in : {word : 1}
    #       if in: {word : word_occur[word] + 1}
        feature_sets.append((word_occur, i[0]))
    #   feature_sets.append((wor_occur, i[0]))
    


#    suffix_fdist = nltk.FreqDist(feature_sets)
#    common_suffixes = [suffix for (suffix, count) in suffix_fdist.most_common(100)]
#
#    def pos_features(word):
#        features = {}
#        for suffix in common_suffixes:
#            features['endswith({})'.format(suffix)] = word.lower().endswith(suffix)
#        return features
#    featuresets = [(pos_features(n), g) for (n,g) in tagged_words]


    training_set = feature_sets[:50] #uses 50 joke to train the classifier
    testing_set= feature_sets[50:]  #uses the other to test result
    classifier = nltk.DecisionTreeClassifier.train(training_set, depth_cutoff=50, support_cutoff=50, entropy_cutoff=0.1)
#classifier = nltk.DecisionTreeClassifier.train(training_set)
    #add informative feature into feature set
    #for feat in classifier.most_informative_features(5):
    #   print(feat)
    #   if feat[1]:
    #       feature.add(feat[0])
    total += (nltk.classify.accuracy(classifier, testing_set))*100 #get percentage
#    accuracy_set = []
#    for word_occur in feature_sets
#        accuracy_set.append(
print("Classifier accuracy percent:", total/500, '\n')
cm = nltk.ConfusionMatrix(classifier, i[3])
print(cm.pretty_format(sort_by_count=True, show_percents=True, truncate=9))
#print("most informative features", feature)
#print(common_suffixes)
#print (classifier.classify(pos_features('')))
#def entropy(labels):
#    freqdist = i[1].freqdist(labels)
#    probs = [freqdist.freq(l) for l in freqdist]
#    return -sum(p * math.log(p,2) for p in probs)
#print(entropy(['', '', '', '']))

