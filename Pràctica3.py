#Practica 3 CRI
#Yassin Nakmouche Sahli M'Ghaiti
#1674585
#Grup G8:30



from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from collections import defaultdict
import numpy as np
import pandas as pd


def open_fitxer():
    df = pd.read_csv("FinalStemmedSentimentAnalysisDataset.csv", delimiter=';')
    df = df.dropna(how='any')  # eliminar files amb valors nuls
    numpy_df = df.to_numpy()
    sentiment_labels = df.iloc[:, 3]

    # Comptem els valors 1 i 0
    num_positius = (sentiment_labels == 1).sum()
    num_negatius = (sentiment_labels == 0).sum()

    # Mostrem el resultat
    print(f"nombre de tweets positius 1: {num_positius}")
    print(f"nombre de tweets negatius0: {num_negatius}")
    
    X = numpy_df[:, 1]  
    Y = numpy_df[:, 3]
    Y = Y.astype('int64')
    return X, Y

def generate_dictionary(X_train, Y_train):
    word_freq = defaultdict(lambda: {'positive': 0, 'negative': 0})
    for text, label in zip(X_train, Y_train):
        for word in text.split():
            if label == 1:
                word_freq[word]['positive'] += 1
            else:
                word_freq[word]['negative'] += 1
    return word_freq

# predecir utilizando Laplace smoothing
def predict(tweet, dictionary, total_pos, total_neg, vocab_size):
    pos_prob = np.log(total_pos / (total_pos + total_neg))
    neg_prob = np.log(total_neg / (total_pos + total_neg))
    
    for word in tweet.split():
        if word in dictionary:
            pos_count = dictionary[word]['positive']
            neg_count = dictionary[word]['negative']
            
            if pos_count > 0:
                pos_prob += np.log(pos_count / total_pos)
            if neg_count > 0:
                neg_prob += np.log(neg_count / total_neg)
        # Si la palabra no está en el diccionario, se ignora.
    
    return 1 if pos_prob > neg_prob else 0
# lectura del dataset
X, Y = open_fitxer()

# configuracio del KFold
k = 5  # Nombre de plegaments
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []  # Per emmagatzemar l'accuracy de cada iteraci�

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # generar diccionari
    dictionary = generate_dictionary(X_train, Y_train)
    
    # Comptar paraules positives i negatives
    total_pos = sum([freq['positive'] for freq in dictionary.values()])
    total_neg = sum([freq['negative'] for freq in dictionary.values()])
    vocab_size = len(dictionary)
    
    # prediccions
    predictions = [predict(tweet, dictionary, total_pos, total_neg, vocab_size) for tweet in X_test]
    
    # Avaluaci�
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)
    print(f"Fold Accuracy Part C: {accuracy}")

# accuracy mitjana
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy across {k} folds Part C: {mean_accuracy}")


###############################  PART B ##############################
print("Part B")

# estrategia 1: ampliar conjunto de entrenamiento
def experiment_strategy_1(X, Y):
    train_sizes = [100, 200, 400]  # Reducido para simplificar
    results = []
    
    for train_size in train_sizes:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=42)
        dictionary = generate_dictionary(X_train, Y_train)
        
        total_pos = sum([freq['positive'] for freq in dictionary.values()])
        total_neg = sum([freq['negative'] for freq in dictionary.values()])
        vocab_size = len(dictionary)
        
        predictions = [predict(tweet, dictionary, total_pos, total_neg, vocab_size) for tweet in X_test[:200]]
        accuracy = accuracy_score(Y_test[:200], predictions)
        results.append((train_size, accuracy, vocab_size))
    
    return results


# çestrategia 2: modificar tamany del diccionari
def experiment_strategy_2(X_train, Y_train, X_test, Y_test):
    dictionary = generate_dictionary(X_train, Y_train)
    sorted_dict = sorted(dictionary.items(), key=lambda item: sum(item[1].values()), reverse=True)
    dictionary_sizes = [5000, 10000, 20000]  # Más pequeño para simplificar
    results = []
    
    for size in dictionary_sizes:
        reduced_dict = dict(sorted_dict[:size])
        
        total_pos = sum([freq['positive'] + 1 for freq in reduced_dict.values()])
        total_neg = sum([freq['negative'] + 1 for freq in reduced_dict.values()])
        vocab_size = len(reduced_dict)
        
        predictions = [predict(tweet, reduced_dict, total_pos, total_neg, vocab_size) for tweet in X_test[:size]]
        accuracy = accuracy_score(Y_test[:size], predictions)
        results.append((size, accuracy))
    
    return results


# estrategia 3: modificar conjunt dentrenamiento amb diccionari fixe
def experiment_strategy_3(X, Y, fixed_dict_size=10000):
    train_sizes = [1000, 2000, 4000]  # Reducido para simplificar
    results = []
    
    for train_size in train_sizes:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=train_size, random_state=42)
        dictionary = generate_dictionary(X_train, Y_train)
        sorted_dict = sorted(dictionary.items(), key=lambda item: sum(item[1].values()), reverse=True)
        reduced_dict = dict(sorted_dict[:fixed_dict_size])
        
        total_pos = sum([freq['positive'] for freq in reduced_dict.values()])
        total_neg = sum([freq['negative'] for freq in reduced_dict.values()])
        vocab_size = len(reduced_dict)
        
        predictions = [predict(tweet, reduced_dict, total_pos, total_neg, vocab_size) for tweet in X_test[:1000]]
        accuracy = accuracy_score(Y_test[:1000], predictions)
        results.append((train_size, accuracy))
    
    return results


# lectura del dataset
X, Y = open_fitxer()

# reduir el conjunte per fer proves rapides
X, Y = X[:100000], Y[:100000]  # Solo 1000 ejemplos para simplificar

# Ejecutar estrategias
results_strategy_1 = experiment_strategy_1(X, Y)
print("estrategia 1: ampliar elconjunt dentrenament", results_strategy_1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=500, random_state=42)
results_strategy_2 = experiment_strategy_2(X_train, Y_train, X_test, Y_test)
print("estrategia 2: modificar mida del diccionari", results_strategy_2)
results_strategy_3 = experiment_strategy_3(X, Y)
print("estrategia 3: modificar conjunt dentrenament amvb diccionari fixe", results_strategy_3)


###############################  PART A ##############################
print("Part A")


# predecir utilizando Laplace smoothing
def predictLaplaceSmoothing(tweet, dictionary, total_pos, total_neg, vocab_size, alpha=1):
    pos_prob = np.log(total_pos / (total_pos + total_neg))
    neg_prob = np.log(total_neg / (total_pos + total_neg))
    
    for word in tweet.split():
        # usar valores por defecto si la palabra no está en el diccionario
        pos_count = dictionary[word]['positive'] if word in dictionary else 0
        neg_count = dictionary[word]['negative'] if word in dictionary else 0
        
        pos_prob += np.log((pos_count + alpha) / (total_pos + alpha * vocab_size))
        neg_prob += np.log((neg_count + alpha) / (total_neg + alpha * vocab_size))
    
    return 1 if pos_prob > neg_prob else 0

X, Y = open_fitxer()

# configuracio del KFold
k = 5  # Nombre de plegaments
kf = KFold(n_splits=k, shuffle=True, random_state=42)

accuracies = []  # Per emmagatzemar l'accuracy de cada iteraci�

for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]
    
    # generar diccionari
    dictionary = generate_dictionary(X_train, Y_train)
    
    # Comptar paraules positives i negatives
    total_pos = sum([freq['positive'] for freq in dictionary.values()])
    total_neg = sum([freq['negative'] for freq in dictionary.values()])
    vocab_size = len(dictionary)
    
    # prediccions
    predictions = [predictLaplaceSmoothing(tweet, dictionary, total_pos, total_neg, vocab_size) for tweet in X_test]
    
    # avaluacio
    accuracy = accuracy_score(Y_test, predictions)
    accuracies.append(accuracy)
    print(f"Fold Accuracy Part A: {accuracy}")

# accuracy mitjana
mean_accuracy = np.mean(accuracies)
print(f"Mean Accuracy across {k} folds Part A: {mean_accuracy}")