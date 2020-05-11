# Importer les données et les départager
# definir une fonction de definition de la notion de vosinage
# definir une fonction de qui renvoie le nombre de voisin de k donnée
# definir une fonction de selection du label

import numpy as np
import math 
import pandas as pd

class k_nn_algorithm:
    
   def __init__(self,given_features,given_results,k):
     """
     """
     assert(k%2 != 0 and k != 0 and k < len(given_features))
     self.__k = k
     self.__data = given_features
     self.__results = given_results
     self.__memory = []
     
   def get_memory(self):
       return self.__memory
    
   def max(dictionnary):
        max = -1
        for key in dictionnary.keys():
            value = dictionnary[key]
            if value > max:
                max = value
        
        return max
    
     
   def euclidian_distance(self,undefined_data):
       number_of_line = len(self.__data)
       for ligne in range(number_of_line):
           distance = 0
           for colonnes in range(undefined_data.shape[0]):
               try:
                   distance += pow(self.__data.iloc[ligne,colonnes] - undefined_data[colonnes],2)
               except AttributeError:
                   distance += pow(self.__data[ligne][colonnes] - undefined_data[colonnes],2)
           
           self.__memory.append((distance,ligne))
       
       self.__memory.sort(key=lambda x: x[0])
       return None
    
   def getLabel(self,undefined_data):
        d = dict()
        
        neighbours = self.getNeighbours(undefined_data)
        for neighbour in neighbours:
            
            try:
                label = self.__results.iloc[neighbour]
            except:
                label = self.__results[neighbour]
                
            if label in d.keys():
                d[label] += 1
            else:
                d[label] = 1
                
        return max(d)
        
        
        
  
  
   def getNeighbours(self,underfined_data):
    """
    retourne le numero de ligne des k plus proches voisins
    """
    self.euclidian_distance(underfined_data)
   
    l = list()
    for couple in range(self.__k):
        l.append(self.__memory[couple][1])
    return l

   def test(self,xtest,ytest):
      nb_test = xtest.shape[0]
      reussi = 0
      for test in range(nb_test):
          current_test = xtest[test]
          current_answer = ytest[test]
          if self.getLabel(current_test) == str(current_answer):
              reussi += 1
      return reussi / nb_test
        


###################################################################


data = pd.read_csv("../Downloads/iris.csv") 

X = data.iloc[:,:data.shape[1]-1]
Y = data.iloc[:,data.shape[1]-1]

    
a = k_nn_algorithm(X,Y,5)
b= np.array([5,3.2,1.2,.2]) #setosa
print("We except a setosa and result is -> " + a.getLabel(b) + " ... Well done")

########################
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

sample = np.random.randint(70000, size=5000)
data = mnist.data[sample]
target = mnist.target[sample]

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(data, target, train_size=0.8)


algo = k_nn_algorithm(xtrain,ytrain,5)
c = xtest[0]
print("the results of my algorithms prediction is : " + algo.getLabel(c))
print("the good answer was " + str(ytest[0])+" well done")
#hesitation car core = 0.2 
test = algo.test(xtest[:10],ytest[:10])

#trop long à l'execution mais est cencé marché
#print("the score is for number prediction is " + test)
#print("the error rate is "+ str(1 - test))

