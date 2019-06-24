import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
stemmer = WordNetLemmatizer()

def SenVal_metric(y_true, y_predi):
    
    output = np.array(y_predi)
    expected = np.array(y_true)
    
    return cosine_similarity(expected, output)

def NLP_pre_processing(title):
    
    documents = []

    for sen in range(0, len(title)):
        # Remove all the special characters
        document = re.sub(r'\W', ' ', str(title[sen]))

        # remove all single characters
        document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    
        # Remove single characters from the start
        document = re.sub(r'\^[a-zA-Z]\s+', ' ', document) 
    
        #Substituing multiplr space with single space
        document = re.sub(r'\s+', ' ', document, flags=re.I)
    
        # Removing prefixed 'b'
        #document = re.sub(r'^b\s+', '', document)
    
        # Converting to Lowercase
        document = document.lower()
    
        # Lemmatization
        document = document.split()
    
        document = [stemmer.lemmatize(word) for word in document]
        document = ' '.join(document)

        documents.append(document)
        
    return documents

def plot_learning_curve(estimator, title, X, y, xlim=None,ylim=None, cv=None, n_jobs=None,
                        scoring = 'neg_mean_squared_error',
                        y_label="MSE",train_sizes=np.linspace(.1, 1.0, 5),
                                   kind='regrassion'):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.xlabel("Training examples")
    plt.ylabel(y_label)
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring=scoring, verbose=2)
    
    if kind == 'regrassion':
        train_scores_mean = -np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = -np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()
        
    else:
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

    
def plot_validation_curve(estimator, title, x_label, X, y, param_name, param_range, ylim=None, cv=None, n_jobs=None):
    
    # Create plot
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel("MSE")
    if ylim is not None:
        plt.ylim(*ylim)
    lw = 2
    
    #Apply Validation Curve
    train_scores, test_scores = validation_curve(estimator, X,
                                                 y, param_name=param_name,
                                                 param_range=param_range,cv=cv,
                                                 scoring='neg_mean_squared_error',
                                                 n_jobs=n_jobs)
  
    # Invert the sign
    train_scores = train_scores* (-1)
    test_scores = test_scores* (-1)
    
    # Calculate mean and standard deviation for training set scores
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
   
    # Plot MSE scores for training and test sets
    plt.semilogx(param_range, train_scores_mean, label="Training MSE",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.2,
                 color="darkorange", lw=lw)

    # Plot MSE bands for training and test sets
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation MSE",
             color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.2,
                 color="navy", lw=lw)

    #Print Plot
    plt.legend(loc="best")
    return plt
    
    
    