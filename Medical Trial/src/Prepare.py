"""
    This file contains the code for the creation of the Datasets: train.csv and test.csv.
    Both files are contained in the dataset directory.
    The file receives two parameters from command line:
    1. The size of the training dataset.
    2. The size of the testing dataset.
"""

__author__      = "Edoardo Oranger"
__version__     = 1.0

import sys
import numpy
import pandas

from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

#! Checks if the number of arguments is 2
if len(sys.argv) != 3:
    sys.exit("Incorrect number of arguments!")


N = int(sys.argv[1]) # The total number of patients, set to 1000 for each half

train_samples = []
train_labels  = []

for i in range(0, 50):
    # The 5% of 65 years old patients or older, hadn't some side effects.
    random_age_older   = randint(65, 100)
    train_samples.append(random_age_older)
    train_labels.append(0)
    
    # The 5% of patients under 65 years old, had some side effects.
    random_age_younger = randint(13, 64)
    train_samples.append(random_age_younger)
    train_labels.append(1)
    
for i in range(50, N):
    # The 95% of 65 years old patients or older, had some side effects.
    random_age_older   = randint(65, 100)
    train_samples.append(random_age_older)
    train_labels.append(1)
    
    # The 95% of patients under 65 years old, had no side effects.
    random_age_younger = randint(13, 64)
    train_samples.append(random_age_younger)
    train_labels.append(0)
    
# Let's convert our lists in numpy arrays
train_samples = numpy.array(train_samples)
train_labels  = numpy.array(train_labels)

# We are going to shuffle them for our Neural Network
train_labels, train_samples = shuffle(train_labels, train_samples)

# Typicaly, data are not normalized, also in our case.
# To normalize them, we are going to use the MinMaxScaler, compressing our data in range [0, 1]
scaler               = MinMaxScaler(feature_range = (0, 1))
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# Now that we have created our dataset, we can export it in the CSV format by using Pandas
data_frame = pandas.DataFrame({'values': scaled_train_samples.reshape(N*2,), 'labels': train_labels})
data_frame.to_csv('../dataset/train/train.csv')

# Let's do the same thing but for test dataset

N = int(sys.argv[2]) # The total number of patients, set to 100 for each half

test_samples = []
test_labels  = []

for i in range(0, int(N*10/100)):
    # The 10% of 65 year old patients or older, has not some side effects.
    random_age_older = randint(65, 90)
    test_samples.append(random_age_older)
    test_labels.append(0)
    
    # The 10% of patients under 65 years old, has some side effects.
    random_age_younger = randint(18, 64)
    test_samples.append(random_age_younger)
    test_labels.append(1)

for i in range(int(N*10/100), N):
    # The 90% of 65 year old patients or older, has some side effects.
    random_age_older = randint(65, 90)
    test_samples.append(random_age_older)
    test_labels.append(1)
    
    # The 90% of patients under 65 years old, has not some side effects.
    random_age_younger = randint(18, 64)
    test_samples.append(random_age_younger)
    test_labels.append(0)
    
# Let's convert our lists in numpy arrays
test_samples = numpy.array(test_samples)
test_labels  = numpy.array(test_labels)

# We are going to shuffle them for our Neural Network
test_labels, test_samples = shuffle(test_labels, test_samples)

# Typicaly, data are not normalized, also in our case.
# To normalize them, we are going to use the MinMaxScaler, compressing our data in range [0, 1]
scaler              = MinMaxScaler(feature_range = (0, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

data_frame = pandas.DataFrame({'values': scaled_test_samples.reshape(N*2,), 'labels': test_labels})
data_frame.to_csv('../dataset/test/test.csv')