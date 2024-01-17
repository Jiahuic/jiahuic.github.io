# Data Preprocessing
We will use Data Preprocessing to learn the basics of Python and its useful packages. 
The data preprocessing is the first step in the data science (machine learning) pipeline.
As we discussed in the first lecture, 
the primary goal of data preprocessing is to transform raw data into a format that can be efficiently and effectively processed by machine learning algorithms.

**How to convert data to numbers?**

## Python Basics and Useful Packages
Python is a general-purpose programming language that is becoming more and more popular for doing data science.
Python is a high-level, interpreted, interactive and object-oriented scripting language.
It can be installed on different operating systems such as Windows, Linux, and Mac, and can be also run on servers such as colab.
You are recommended to use [Anaconda](https://www.anaconda.com/) to install python and its packages or run the jupyter notebook directly on [colab](https://colab.google).

``` python
# This is the first in lab 1
# print the platform, OS, environment, cpu, memory, and username information
import platform, os, sys, getpass, psutil
print('Platform:', platform.platform())
print('OS:', os.name)
print('Environment:', sys.executable)
print('CPU:', platform.processor())
print('Memory:', psutil.virtual_memory().total / (1024.0 ** 3), 'GB')
print('Username:', getpass.getuser())
```

### Python Environment
Python environment is a context in which a Python program runs. An environment consists of an interpreter and any number of installed packages. See [Python Environment](https://docs.python.org/3/tutorial/venv.html).
Using `conda` to create and control your Python environment is a good choice. See [Managing environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html).

We will use the following python packages in this lecture:
1. [python](https://docs.python.org/3/library/index.html): the official python documentation.
1. [numpy](https://numpy.org/): a package for scientific computing with Python.
1. [pandas](https://pandas.pydata.org/): a fast, powerful, flexible and easy to use open source data analysis and manipulation tool.
1. [scikit-learn](https://scikit-learn.org/stable/): a machine learning library for Python.
1. [mnist](https://pypi.org/project/python-mnist/): a package for reading MNIST dataset.
1. [matplotlib](https://matplotlib.org/): a comprehensive library for creating static, animated, and interactive visualizations in Python.

### 1. `python` -- The official python documentation
In this course, we will use python 3.12.1 and its packages. For certen packages, we will use the older version of python.
The official python documentation is a good place to learn the basic python syntax and its packages.

#### - Python function
A function is a block of organized, reusable code that is used to perform a single, related action. Functions provide better modularity for your application and a high degree of code reusing. As you already know, Python gives you many built-in functions like `print()`, etc. but you can also create your own functions. These functions are called user-defined functions. See [Defining Functions](https://docs.python.org/3/tutorial/controlflow.html#defining-functions).
``` python
# define a function
def function_name(parameters):
    """docstring"""
    statement(s)
    return [expression]
# call a function
function_name(parameters)
```

#### - [Bulti-in Function](https://docs.python.org/3/library/functions.html): a list of built-in functions in python. The following functions are commonly used in data preprocessing.
---
- `abs()`: Return the absolute value of a number. The argument may be an integer or a floating point number. If the argument is a complex number, its magnitude is returned.
- `max()` and `min()`: Return the largest or the smallest item in an iterable or the largest of two or more arguments. See [Built-in Functions](https://docs.python.org/3/library/functions.html#max).
- `pow()`: Return base to the power exp; if mod is present, return base to the power exp, modulo mod (computed more efficiently than pow(base, exp) % mod). The two-argument form pow(base, exp) is equivalent to using the power operator: base, exp. See [Built-in Functions](https://docs.python.org/3/library/functions.html#pow).
---
- `dict()`: Create a new dictionary. The dict object is the dictionary class. See [Mapping Types — dict](https://docs.python.org/3/library/stdtypes.html#mapping-types-dict).
- `list()`: Return a list whose items are the same and in the same order as iterable‘s items. iterable may be either a sequence, a container that supports iteration, or an iterator object. See [list](https://docs.python.org/3/library/stdtypes.html#list).
---
- `enumerate()`: Return an enumerate object. The enumerate object yields pairs containing a count (from start, which defaults to zero) and a value yielded by the iterable argument. See [enumerate](https://docs.python.org/3/library/functions.html#enumerate).
- `len()`: Return the length (the number of items) of an object. The argument may be a sequence (such as a string, bytes, tuple, list, or range) or a collection (such as a dictionary, set, or frozen set). See [Built-in Functions](https://docs.python.org/3/library/functions.html#len).
- `next()`: Retrieve the next item from the iterator by calling its __next__() method. If default is given, it is returned if the iterator is exhausted, otherwise StopIteration is raised. See [Built-in Functions](https://docs.python.org/3/library/functions.html#next).
- `range()`: Rather than being a function, range is actually an immutable sequence type, as documented in Ranges and Sequence Types — list, tuple, range. See [Built-in Functions](https://docs.python.org/3/library/functions.html#func-range).
---
- `open()`: Open file and return a corresponding file object. If the file cannot be opened, an OSError is raised. See [Built-in Functions](https://docs.python.org/3/library/functions.html#open).
- `print()`: Print objects to the text stream file, separated by sep and followed by end. sep, end, file and flush, if present, must be given as keyword arguments. See [Built-in Functions](https://docs.python.org/3/library/functions.html#print).
---

``` python
# initialize a list 
my_list = []
# append an item to the list 
my_list.append(1)
my_list.append(2)
my_list.append(3)
print(my_list, len(my_list))

# initialize a dictionary
my_dict = {}
# add an item to the dictionary
my_dict['a'] = 1 
my_dict['b'] = 2 
my_dict['c'] = 3 
print(my_dict, len(my_dict.keys()), len(my_dict.values()))
```

#### - [Built-in Types](https://docs.python.org/3/library/stdtypes.html): a list of built-in types in python.
Python uses the common data types such as `int`, `float`, `str`, `list`, `dict`, and `bool`.
The following description is based on `list` type. `list` is a mutable sequence type that can be written as a list of comma-separated values (items) between square brackets.
``` python
# list
list = [1, 2, 3, 4, 5]
print(list)
```
`list` can contains different types of data, such as having a list of lists or a list of `float` and `int` types in the same list.
``` python
# list of lists
list = [[1, 2, 3], [4, 5, 6]]
print(list)
# list of float and int
list = [1, 2, 3, 4, 5, 6.0]
print(list)
```
The `list` type has some useful methods such as `append()`, `extend()`, `insert()`, `remove()`, `pop()`, `clear()`, `index()`, `count()`, `sort()`, and `reverse()`.
``` python
list = [1, 2, 3, 4, 5]
# list append
list.append(6)
print(list)
# list extend
list.extend([7, 8, 9])
print(list)
# list insert
list.insert(0, 0)
print(list)
# list remove
list.remove(0)
print(list)
# list index
print(list.index(1))
```

Another part is about the **Boolean Operations**. The Boolean type is a subtype of the integer type, and Boolean values behave like the values 0 and 1, respectively, in almost all contexts, the exception being that when converted to a string, the strings "False" or "True" are returned, respectively. See [Boolean Operations](https://docs.python.org/3/library/stdtypes.html#boolean-operations-and-or-not).
``` python
# if statement
if True:
    print('True')
else:
    print('False')

# if statement with and, or, not
flag = True
if not flag:
    print('False')
elif flag and not flag:
    print('False')
else:
    print('True')
```

#### - For Loop
The `for` statement is used to iterate over the elements of a sequence (such as a string, tuple or list) or other iterable object. Iterating over a sequence is called traversal. See [for statement](https://docs.python.org/3/reference/compound_stmts.html#for).
``` python
# for loop
for i in range(10):
    print(i)
```

#### - Useful Standard Modules (Packages)
We will use the following packages in this lecture:
- [os](https://docs.python.org/3/library/os.html): Miscellaneous operating system interfaces.
- [sys](https://docs.python.org/3/library/sys.html): System-specific parameters and functions.
- [math](https://docs.python.org/3/library/math.html): Mathematical functions.
- [random](https://docs.python.org/3/library/random.html): Generate pseudo-random numbers.
- [time](https://docs.python.org/3/library/time.html): Time access and conversions.
- [re](https://docs.python.org/3/library/re.html): Regular expression operations.
- [argparse](https://docs.python.org/3/library/argparse.html): Parser for command-line options, arguments and sub-commands.

``` python
import os, sys, math, random, time, re, argparse
# generate a random number
print(random.random())
# get the current time 
print(time.time())
# get the current working directory
print(os.getcwd())
# get the current python version
print(sys.version)

# split a string
my_str = '1,2,3,4,5|6,7,8,9,10'
print(my_str.split(','))
print(my_str.split('|'))
print(re.split(',|\|', my_str))
```


### 2. `numpy` -- Numerical Python
The `numpy` package is the core library for scientific computing in Python. Some of its roles can be replaced by `pytorch` or `tensorflow` when consider the deep learning numerical network. We will go back to `pytorch` by the end of this semester.
The `numpy` package provides the `ndarray` object for efficient storage and manipulation of dense data arrays in Python. The `numpy` array is a powerful N-dimensional array object which is in the form of rows and columns. We can initialize `numpy` arrays from nested Python lists, and access elements using square brackets.
``` python
import numpy as np
# create a numpy array
a = np.array([1, 2, 3])
print(a)
# create a 2d numpy array
b = np.array([[1, 2, 3], [4, 5, 6]])
print(b)
# create a 3d numpy array
c = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11 ,12]]])
print(c)
```

The `numpy` can do the basic math operations such as `+`, `-`, `*`, `/`, and `**`.
``` python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
# add
print(a + b)
# subtract
print(a - b)
# multiply
print(a * b)
# divide
print(a / b)
# power
print(a ** b)
```

**NOTE**: The `numpy` array is different from the `list` type. The `numpy` array is a fixed size array while the `list` type is a dynamic array. The `numpy` array is faster than the `list` type.
``` python
import numpy as np
# add two lists
a = [1, 2, 3]
b = [4, 5, 6]
print(a + b)
# add two numpy arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a + b)
```

The `numpy` package provides some useful functions such as `np.zeros()`, `np.ones()`, `np.full()`, `np.eye()`, `np.random.random()`, `np.arange()`, `np.linspace()`, `np.reshape()`, `np.transpose()`, `np.concatenate()`, `np.vstack()`, and `np.hstack()`.
``` python
import numpy as np
# create a numpy array with all zeros
a = np.zeros((2, 2))
print(a)
# create a numpy array with all ones
b = np.ones((2, 2))
print(b)
# create a numpy array with all full
c = np.full((2, 2), 7)
print(c)
# create a numpy array with identity matrix
d = np.eye(2)
print(d)
# create a numpy array with random values
e = np.random.random((2, 2))
print(e)
# create a numpy array with a range
f = np.arange(10)
print(f)
# create a numpy array with evenly spaced values
g = np.linspace(0, 1, 5)
print(g)
# create a numpy array with reshape
h = np.arange(10).reshape((2, 5))
print(h)
# create a numpy array with transpose
i = np.arange(10).reshape((2, 5))
print(i)
print(i.T)
# create a numpy array with concatenate
j = np.array([[1, 2, 3], [4, 5, 6]])
k = np.array([[7, 8, 9], [10, 11, 12]])
print(np.concatenate([j, k], axis=0))
print(np.concatenate([j, k], axis=1))
# create a numpy array with vstack
print(np.vstack([j, k]))
# create a numpy array with hstack
print(np.hstack([j, k]))
```

The `numpy.linalg` package provides some useful functions such as `np.linalg.inv()`, `np.linalg.det()`, `np.linalg.eig()`, `np.linalg.svd()`, `np.linalg.solve()`, and `np.linalg.lstsq()`.
``` python
import numpy as np
# create a numpy array
a = np.array([[1, 2], [3, 4]])
print(a)
# compute the inverse of a matrix
print(np.linalg.inv(a))
# compute the determinant of a matrix
print(np.linalg.det(a))
# compute the eigenvalues and right eigenvectors of a square array
print(np.linalg.eig(a))
# compute the singular value decomposition
print(np.linalg.svd(a))
# solve a linear matrix equation
b = np.array([1, 2])
print(np.linalg.solve(a, b))
# compute least-squares solution to equation
print(np.linalg.lstsq(a, b))
```

### 3. `pandas` -- Python Data Analysis Library
Pandas is a powerful and widely-used Python library for data manipulation and analysis. It offers data structures and operations for manipulating numerical tables and time series, making it a crucial tool for data science and related fields.

#### - `DataFrame` 
A two-dimensional, size-mutable, potentially heterogeneous tabular data. Data structure also contains labeled axes (rows and columns). Arithmetic operations align on both row and column labels. Can be thought of as a dict-like container for Series objects. The primary pandas data structure. See [DataFrame](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html).
``` python
import pandas as pd
# create a pandas dataframe
df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
print(df)
# create a pandas dataframe from numpy array
import numpy as np
df = pd.DataFrame(np.random.rand(3, 2), columns=['col1', 'col2'])
print(df)
# create a pandas dataframe from list
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['col1', 'col2', 'col3'])
print(df)
```
To load csv files into a pandas dataframe, we can use the `pd.read_csv()` function.
Access the dataset of a pandas dataframe.
``` python
import pandas as pd
# load csv file into a pandas dataframe
df = pd.read_csv('data.csv')
print(df)
# access the dataset of a pandas dataframe
print(df.values)
```

#### - `Series`
One-dimensional ndarray with axis labels (including time series). Labels need not be unique but must be a hashable type. The object supports both integer- and label-based indexing and provides a host of methods for performing operations involving the index. Statistical methods from ndarray have been overridden to automatically exclude missing data (currently represented as NaN). See [Series](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.html).
``` python
import pandas as pd
# create a pandas series
s = pd.Series([1, 2, 3, 4, 5])
print(s)
# create a pandas series with index
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)
# create a pandas series from numpy array
import numpy as np
s = pd.Series(np.random.rand(5))
print(s)
# create a pandas series from list
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
print(s)
```

### 4. `scikit-learn` -- Dataset loading utilities
The `sklearn.datasets` package provides the most popular machine learning methods basically for classification, regression, and clustering.
In this course, we will discuss most of the methods in this package, code them from scratch and compare the results with the `sklearn` package.
Those methods and their useage of the `sklearn` package will be discussed when we discuss the corresponding machine learning methods.
Here, at the beginning of this course, we will discuss the `sklearn.datasets` package and its utilities.
The `sklearn.datasets` package provides some utilities to load popular datasets [link](https://scikit-learn.org/stable/datasets.html).
The function includes some small toy datasets and real world datasets
- [Iris plant dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#iris-plants-dataset): The iris dataset is a classic and very easy multi-class classification dataset.
- [Diabetes dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#diabetes-dataset): En baseline variables, age, etc. were obtained for each of n=442 diabetes patients.
- [Digits dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#digits-dataset): Optdigits dataset.
- [Wine recognition dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-recognition-dataset): The data is the results of a chemical analysis of wines grown in the same region in Italy by three different cultivators.
- [California Housing dataset](https://scikit-learn.org/stable/datasets/toy_dataset.html#california-housing-dataset): This dataset was obtained from the StatLib repository. See [link](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html).
``` python
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_boston, load_breast_cancer, load_linnerud, load_sample_image, load_sample_images, load_svmlight_file, load_files

# Load the dataset
iris = load_iris()
# To view the features (the measurements for each flower)
print(iris.data)
# To view the target values (the species of each flower)
print(iris.target)
# To view the names of the target species
print(iris.target_names)
# To view the feature names
print(iris.feature_names)
# Description of the dataset
print(iris.DESCR)
```

### 5. `mnist` -- MNIST dataset loader
The `mnist` package provides a function to load the MNIST dataset. The MNIST dataset is a dataset of handwritten digits. It has 60,000 training samples, and 10,000 test samples. Each image is represented by 28x28 pixels, each containing a value 0 - 255 with its grayscale value. The MNIST dataset is one of the most common datasets used for image classification and accessible from many different sources. See [MNIST dataset](http://yann.lecun.com/exdb/mnist/).
``` python
# load from mnist dataset: python-mnist
# train-images-idx3-ubyte.gz:  training set images (9912422 bytes)
# train-labels-idx1-ubyte.gz:  training set labels (28881 bytes)
# t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes)
# t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
from mnist import MNIST
# Initialize the dataset
mndata = MNIST('./')
# Load the dataset into memory (this will search the four files above)
training_images, training_labels = mndata.load_training()
testing_images, testing_labels = mndata.load_testing()
```

<span style="color:red">**Question:**</span> How to load an RGB image dataset?

``` python
from PIL import Image
import numpy as np

# Load the image
image = Image.open('figures/lec2_pic1.png')

# Split the image into its R, G, B components
r, g, b, a = image.split()

# Create a new image with the same size but only with the red component
r_image = Image.merge("RGB", (r, Image.new('L', r.size), Image.new('L', r.size)))

# Similarly, create images for the green and blue components
g_image = Image.merge("RGB", (Image.new('L', g.size), g, Image.new('L', g.size)))
b_image = Image.merge("RGB", (Image.new('L', b.size), Image.new('L', b.size), b))

# Convert the R channel to a numpy array
r_array = np.array(r)

# Print the matrix
print(r_array)

# Save or display the images
r_image.show() # This will display the image
# g_image.show()
# b_image.show()

# Optionally, save the images
# r_image.save('path_to_save_red_component.jpg')
# g_image.save('path_to_save_green_component.jpg')
# b_image.save('path_to_save_blue_component.jpg')
```

### **NOTE**: Other useful data preprocessing packages
#### Text analysis and natural language processing
- [NLTK](https://www.nltk.org/): a leading platform for building Python programs to work with human language data.
- [spaCy](https://spacy.io/): a free, open-source library for advanced Natural Language Processing (NLP) in Python.
- 20 Newsgroups dataset: a collection of approximately 20,000 newsgroup documents, partitioned (nearly) evenly across 20 different newsgroups.
- [tensorflow-datasets](https://www.tensorflow.org/datasets): a collection of datasets ready to use with TensorFlow.

**For example**:
``` python
import pandas as pd    # to load dataset
data = pd.read_csv('../datasets/IMDB.csv')
# print(data)
```
<hr>
<b>Stop Word</b> is a commonly used words in a sentence, usually a search engine is programmed to ignore this words (i.e. "the", "a", "an", "of", etc.)

<i>Declaring the english stop words</i>
``` python
import nltk
from nltk.corpus import stopwords   # to get a collection of stopwords
custom_path = '../datasets/'

# Append your custom path to the NLTK data path
nltk.data.path.append(custom_path)

nltk.download('stopwords', download_dir=custom_path)
english_stops = set(stopwords.words('english'))
```
<hr>
<b>Load and Clean Dataset</b>
In the original dataset, the reviews are still dirty. There are still html tags, numbers, uppercase, and punctuations. This will not be good for training beside loading the dataset using <b>pandas</b>, I also pre-process the reviews by removing html tags, non alphabet (punctuations and numbers), stop words, and lower case all of the reviews.
``` python
x_data = data['review']       # Reviews/Input
y_data = data['sentiment']    # Sentiment/Output
# PRE-PROCESS REVIEW
x_data = x_data.replace({'<.*?>': ''}, regex = True)          # remove html tag
x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)     # remove non alphabet
x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
x_data = x_data.apply(lambda review: [w.lower() for w in review])   # lower case
```
<hr>
<b>Tokenize</b> A Neural Network only accepts numeric data, so we need to encode the reviews. We can do this by using Tokenizer from Keras. Tokenizer will tokenize the reviews and convert it into sequence of numbers. The numbers represent the index of the word in the dictionary. The dictionary is created by the Tokenizer. The dictionary is created based on the frequency of the word in the dataset. The most frequent word will have the lowest index and vice versa.
``` python
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(num_words=10000)    # num_words is the number of words to keep based on word frequency
tokenizer.fit_on_texts(x_data)            # fit tokenizer to our training text data
x_data = tokenizer.texts_to_sequences(x_data)  # convert our text data to sequence of numbers
```

#### Image processing
- [Pillow](https://pillow.readthedocs.io/en/stable/): the friendly PIL fork (Python Imaging Library).
- [OpenCV](https://opencv.org/): a library of programming functions mainly aimed at real-time computer vision.
- [scikit-image](https://scikit-image.org/): a collection of algorithms for image processing.

#### Time series analysis
- [statsmodels](https://www.statsmodels.org/stable/index.html): a Python module that provides classes and functions for the estimation of many different statistical models, as well as for conducting statistical tests, and statistical data exploration.
- [Airline Passengers dataset](https://www.kaggle.com/rakannimer/air-passengers): The classic Box & Jenkins airline data. Monthly totals of international airline passengers, 1949 to 1960.
- [Stock dataset](https://www.kaggle.com/borismarjanovic/price-volume-data-for-all-us-stocks-etfs): This dataset includes historical stock prices (last 5 years) for all companies currently found on the S&P 500 index.
- [Electricity dataset](https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014): This data set contains the electricity consumption of 370 customers from 2011-2014 and is taken from the UCI Machine Learning Repository.

#### Reinforcement learning
- [OpenAI Gym](https://gym.openai.com/): a toolkit for developing and comparing reinforcement learning algorithms.
- [Atari 2600 games](https://gym.openai.com/envs/#atari): a collection of Atari 2600 games that are compatible with OpenAI Gym.

## Data preprocessing for structured data
Structured data is highly organized and easily searchable in databases and spreadsheets, typically in tabular form with rows and columns.
1. **Data cleaning**: Fill in missing values, smooth noisy data, identify or remove outliers, and resolve inconsistencies.
    - **Handling missing values**: Fill missing values using strategies like mean/mode substitution, forward fill, backward fill, or removal of rows/columns.
    - **Removing duplicate data**: Remove duplicate data based on a subset of columns or all columns.
    - **Filtering outliers**: Detect and handle outliers that can affect the analysis (using statistical techniques like Z-scores, IQR)
2. **Data transformation**: Discretize continuous features, handle categorical features, and convert strings to numbers.
    - **Normalization and Standardization**: Rescale data to a standard range or distribution (e.g., Min-Max scaling, Z-score normalization).
    - **Encoding categorical data**: Convert categorical data into numerical form using techniques like one-hot encoding or label encoding.
3. **Data reduction**: Reduce the volume but retain critical information.
    - **Feature selection**: Select a subset of relevant features (columns) for building robust learning models.
    - **Feature extraction**: Derive new features from existing ones that are more compact, informative, and easy to understand (e.g., PCA, tSNE, UMAP).
4. **Feature Engineering**: 
    - **Creating new features**: Derive new meaningful features from existing data (e.g. TDA).
    - **Feature selection**: Select the most relevant features for the model (feature ranking -- random forest).

## Data preprocessing for unstructured data
Unstructured data is not organized in a predefined way and includes formats like text, images, and audio.
1. **Text data**:
    - **Tokenization**: Breaking text into words, phrases, symbols, or other meaningful elements.
    - **Text cleaning**: Removing irrelevant characters, such as special symbols, numbers, and punctuation.
    - **Stop word removal**: Eliminating common words that add no significant value.
    - **Stemming and lemmatization**: Reducing words to their root form or lemma.
    - **Vectorization**: Converting text to numerical format using techniques like Bag of Words, TF-IDF.

2. **Image Data**:
    - **Image resizing and scaling**: Standardizing image sizes and pixel values. **Color space conversion**: Converting images to grayscale or different color spaces if required.
    - **Image augmentation**: Creating altered versions of images to expand the dataset (e.g., flipping, rotating).
    - **Noise reduction**: Applying filters to reduce noise in images.

3. **Audio data**:
    - **Sampling**: Converting continuous audio signals into discrete values.
    - **Noise reduction**: Removing background noise.
    - **Feature extraction**: Extracting features like Mel-Frequency Cepstral Coefficients (MFCCs), Zero-Crossing Rate, Chroma.

4. **Graph data**:
   - **Protein structure**: Protein structure prediction.
   - **Power system**: Power system state estimation.

**NOTE**: Now, this is a good time to think about your dataset.
