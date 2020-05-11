
######################   PARTIE 1   ##########################################

#import des modules nécessaires

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression

# 1 - Dataset

x , y = make_regression(n_samples=100,n_features=1, noise=10)

X = np.hstack( ( x,np.ones((x.shape[0],1)) ) )
Y = y.reshape(y.shape[0],1)
theta = np.random.randn(2,1) #matrice 0 = [a,b] en ligne
 
# 2 - modèle f(x) = ax + b sous forme matriciel
    #Pour des modèle polynomiaux, il suffit juste de modifié la matrice X et theta(add c)
    # f(x) ax^2 + bx + c -> X -> [x2,X,1]


def model(x,theta):
    """
    Renvoie une matrice avec les coordonnées de tout les points
    après avoir appliqué f(x) la fonction du modèle
    """
    return x.dot(theta)


# 3 - fonction de cout - qui permet de juger de l'adéquation du modèle

def cost(x, y, theta):
    """
    Renvoie la valeur du risque empirique ici basé sur
    la distance euclidienne
    """
    m = x.shape[0]
    return (1/2 * m) * np.sum( (model(x, theta) - y)**2)

# 4 - Minimisation - recherche des paramètres optimaux afin que le modèle soit le plus adéquat


def gradient(x,y,theta):
    """
    La fonction du cout est en x**2 donc elle admet un minimum global,
    on va convergé d'un movement à travers cette fonction vers ce optimum
    
    Calcul la dérivé partielle par rapport au composant de theta
    renvoie une matrice contenant les valeurs des dérivé par
    rapport au paramètre theta de la fonction cout 
    """
    m = x.shape[0]
    return (1/m)* x.T.dot(model(x,theta) - y)
    
def gradient_descent(x,y,theta,learning_rate,nb_iteration=1000):
     """
     Fait converger les composantes de theta vers le minimum global 

     Modifie la valeur de theta en effectuant le gradient autant de fois
     que le nombre diteration donnée avec un déplacement de learning_rate
     """
     #peut etre optimisé, si theta ne change gradient est nul break
     for i in range(nb_iteration):
         theta = theta - (learning_rate * gradient(x,y,theta))
     return theta
    

# Final - Fonction de regression Linéaire

def Linear_regression(x,y,theta,learning_rate,nb_iteration=700):
    """
    Determine le couple optimal de theta après nb_itérations
    puis retourne le model avec ses paramètres
    """
    theta_final = gradient_descent(x,y,theta,learning_rate,nb_iteration)
    return model(x,theta_final)

def learning(x,y):
    """
     Version en utilisant la méthode des moindres carrés
     soit trouvé les composantes de theta tels que le gradient est nulle
    """
    return np.linalg.inv(x.T.dot(x)).dot(x.T).dot(y)
    



def affiche(x_debut,x_modif,y,theta,learning_rate,nb_iteration):
    
    model1 = Linear_regression(x_modif,y,theta,learning_rate,nb_iteration)
    model2 = model(x_modif,learning(x_modif,y))
    
    plt.title("Linear regression")
    plt.scatter(x_debut,y,color="grey",label="data")
    plt.plot(x_debut,model1,c="black",label="gradient descent method")
    plt.plot(x_debut,model2,'ro',c="red",label="least square method")
    plt.legend()
    plt.show()
    
    
    


# main_1
affiche(x,X,Y,theta,0.01,1000)

######################   PARTIE 2   ##########################################

from sklearn.model_selection import train_test_split
import pandas as pd

# Traitement des données
init_data = pd.read_csv("../Downloads/house_data.csv")
init_data = init_data[init_data['surface'] < 200]
init_data = init_data[init_data['price'] < 7000]
data2 = init_data.iloc[:,1:]
target2 = init_data.iloc[:,0]


xtrain, xtest, ytrain, ytest = train_test_split(data2, target2, train_size=0.8)
xtrain = np.hstack((xtrain,np.ones((xtrain.shape[0],1))))
ytrain = ytrain.values.reshape(ytrain.shape[0],1)
theta1 = np.random.randn(3,1) 

# Learning part

"""
FINIR SI JAI LE TEMPS
model_1 = model(xtrain,learning(xtrain,ytrain))
plt.title("Multiple Linear regression")
plt.scatter(xtrain[:,0],ytrain,color="grey",label="data")
plt.plot(xtrain[:,1],model_1,'ro',c="black",label="gradient descent method")
plt.legend()
plt.show()
"""


#plt.scatter(xtrain[:,1],ytrain)
#plt.show(block=False)