# Google 1998 - Information Retrieval Project

*By Quentin Churet, Alexis Tacnet, Domitille Prevost at CentraleSupélec.*

This project is a search engine with a good looking terminal interface and some queries to test our search algorithm. This algorithm is based on the TW-IDF described in [this paper](https://frncsrss.github.io/papers/rousseau-cikm2013.pdf), where we compute the link between words in a fixed window.

[![screenshot](https://raw.githubusercontent.com/qchuchu/google-1998/master/assets/screenshot.png)](https://raw.githubusercontent.com/qchuchu/google-1998/master/assets/screenshot.png)

## Setting up the project

### Creating a virtual environment

After cloning the repository, it is highly recommended to install a virtual environment (such as virtualenv) or Anaconda to isolate the dependencies of this project with other system dependencies.

To install virtualenv, simply run:

```
$ pip install virtualenv
```

Once installed, a new virtual environment can be created by running:

```
$ virtualenv venv
```

This will create a virtual environment in the venv directory in the current working directory. To change the location and/or name of the environment directory, change venv to the desired path in the command above.

To enter the virtual environment, run:

```
$ source venv/bin/activate
```

You should see (venv) at the beginning of the terminal prompt, indicating the environment is active. Again, replace venv with your desired directory name.

To get out of the environment, simply run:

```
(venv) $ deactivate
```

### Installing Dependencies

While the virtual environment is active, install the required dependencies by running:

```
(venv) $ pip install -r requirements.txt
```

This will install all of the dependencies at specific versions to ensure they are compatible with one another.

### Installing NLTK libs

We use NLTK as our package for word libraries, and you may need to install some datasets:

```
(venv) $ python -m nltk.downloader stopwords wordnet
```

### Install the data

This search engine was built to work on the CS276 data by Standord Education. The dataset can be found [here](http://web.stanford.edu/class/cs276/pa/pa1-data.zip) and needs to be installed in the `./data/cs276` folder.

### Launching the interface !

We provide a good looking interface for your queries, just by running:

```
(venv) $ python interface.py
```

> The first time you launch our search engine, we will compute the indexes. This may takes some time.

You can also specify the count of results that it will display, like this:

```
(venv) $ python interface.py --count 20
```

### Test the sample queries

In order to test our search algorithm, we have in the `dev_*` folders sample queries and their expected output. You can compute their output from our algorithm by running:

```
(venv) $ python search_engine.py
```

And then you can compute the accuracies of our outputs (percentage of expected output that we got right):

```
(venv) $ python show_accuracy.py
```

## Technical description

### Preprocessing of documents

The documents are preprocessed in the indexing pipeline. In a first time removed the English stopwords, but we kept them in the final version for more accuracy, as the performance for indexing and query time is quite the same.
We then lemmatize the tokens with the WordNet network.

### Algorithm used

We used a TW-IDF where the goal is to compute the number of times a word is linked to a new word, in contrary of the TF-IDF where we only care about the frequency of a word in a document.
