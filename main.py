import nltk
import csv
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from string import punctuation
import random
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
with open(';/⁨Users⁩/cecilia⁩/⁨Cecilia⁩/大一啦⁩/VIPHumorG⁩/⁨FindC⁩/⁨ComedianIdentifier/⁩textfiles/Final CSV  - Sheet1.csv') as file:
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
for counter in range(100):  #train and test 100 times
    random.shuffle(tokenized_data) #shuffle the data so that we get a more accurate result
    feature_sets = [] #actual feature set
    for i in tokenized_data:
        #i[3] is the actual joke, i[0] is name of comedian
        #data cleaning and creating feature dictionary, True means we want this word to be used
        word_feature = dict([(word.lower(), True) for word in i[3] 
        if word not in punctuation 
        and word.lower() not in ['trump', 'obama',  'laughter', 'cheering', 'applause',]
        and word != '``' 
        and word != "''" 
        and word != '""' 
        and word.lower() not in stop_words])
        feature_sets.append((word_feature, i[0])) #feature_set element: (dict of {word, True}, comedian_name)

    training_set = feature_sets[:50] #uses 50 joke to train the classifier
    testing_set= feature_sets[50:]  #uses the other to test result
    classifier = nltk.NaiveBayesClassifier.train(training_set)
    #add informative feature into feature set
    for feat in classifier.most_informative_features(5):
        print(feat)
        if feat[1]:
            feature.add(feat[0])
    total += (nltk.classify.accuracy(classifier, testing_set))*100 #get percentage
print("Classifier accuracy percent:", total/100, '\n')
print("most informative features", feature)
