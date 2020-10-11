import random 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np 

def sigmoide(x):
    """
    x: int o np.array
        si es un arreglo aplica la función a cada entrada

    Regresa el valor de la función sigmoide en x 
    
    """
    return 1/(1 + np.exp(-x))


def regresion_logistica(X, theta):
    """
    X: np.array
        Arreglo de m ejemplos de entrenamiento (m, d+1)
        cada uno con el término de sesgo como primer entrada


    theta: np.array de (d+1,1)
        vector de parámetros 

    """

    return sigmoide(X@theta)


def costo(y_hat, y):
    """
    y_hat: np.array
        predicciones (m,) (probabilidad de la clase positiva) 
    m número de jemplos
    y: np.array
        etiquetas (m,)
    
    Regresa el costo sobre las predicciones. Número positivo
    """
    m = y.shape[0]

    J  = -(1.0/m) *(y.T@np.log(y_hat)) + (1-y.T)@np.log(1-y_hat)

    return J 

def descenso_gradiente(X,y, theta, alpha, iteraciones):
    """
    X: np.array (m,d+1)
        conjunto de entrenamiento
        d número de características 
    y: np.array (m,)
        etiquetaso respuestas (0 o 1)
    theta: np.array (d+1,) 
        parámetros del modelo
    alpha: float
        tasa de aprendizaje, número positivo
    iteraciones: int 
        número de iteraciones del algoritmo 

    Regresa: El costo (J) y el vector de parámetros theta
    """

    # Número de ejemplos
    m = X.shape[0]

    for i in range(iteraciones):
        y_hat  = regresion_logistica(X,theta) # Predicción (logits, aun no son clases 0  o 1)

        # Calcula el costo. Compara las predicciones y_hat con las respuestas y
        J = costo(y_hat, y)

        # Descenso por el gradiente
        theta  = theta -alpha* (1/m)*X.T@(y_hat-y)
    

    return J, theta



def prueba(X,y,theta):
    """
    X: np.array (m,d+1)
        conjunto de prueba
        d número de características 
    y: np.array (m,)
        etiquetaso respuestas (0 o 1)
    theta: np.array (d+1,) 
        parámetros del modelo
    """

    m = y.shape[0]

    # Realiza predicciones en el conjunto de prueba
    y_hat = regresion_logistica(X,theta)
    # Transforma probabilidades en clases (0 o 1)
    y_hat[y_hat>0.5] = 1 
    y_hat[y_hat<=0.5] = 0

    precision = (1.0)/m*np.sum(y_hat == y)


    return precision


def datos():

    n = 500
    d = 3 # Número de dimensiones en cada ejemplo

    examples = []

    for i in range(n):

        pos = np.random.normal(loc = 0, scale = 1, size=(1,d) )
        neg = np.random.normal(loc = -2, scale = 3, size=(1,d) )

        examples.append((pos,1))
        examples.append((neg,0))

    # Mezcla aleatoriamente los ejemplos
    random.shuffle(examples)

    # Crea X,y 
    X = np.array([ejemplo[0] for ejemplo in examples])
    X = np.squeeze(X)
    m,n  = X.shape
    # Añade el termino de sesgo (una columna de unos al inicio de X)
    bias = np.ones((m,1))
    X  = np.append(bias, X, axis = 1 )

    y = np.array([ejemplo[1] for ejemplo in examples])

    return X,y 



def visualiza_datos(X,y):
    """
    X: np.array (m,d+1) m número de ejemplos, d número de características
    y: np.array (m,) 
    """

    fig = plt.figure()
    ax3D = fig.add_subplot(111, projection='3d')
    #ax3D.scatter(X[1], X[2], X[3], s=10,  marker='o')  
    m,n = X.shape 

    for i in range(m):
        x,clase  = X[i], y[i]

        marker = '^'
        color = 'b'

        if clase == 0:
            marker = 'o'
            color = 'r'
        
        ax3D.scatter(x[1],x[2],x[3],s = 30, c = color, marker = marker)

    plt.show()


if __name__ == "__main__":
    import math 
    X,y  = datos()

   

    m,e = X.shape
    # fracción que va al conjunto de entrenamiento,
    # lo restante va al conjunto de prueba
    fracc = 0.8
    n_ent  = int(m*fracc)
    n_prueba = math.ceil(m*(1-fracc))

    X_ent, X_prueba = X[:n_ent],X[n_ent:]
    y_ent, y_prueba = y[:n_ent], y[n_ent:]
    print("Tamaño conjunto de entrenamiento {} y conjunto de prueba {}".format(
        n_ent,
        n_prueba))

    # Visualiza los datos 
    #visualiza_datos(X_ent,y_ent)

    theta = np.zeros((e,))
    alpha = 0.3
    iteraciones = 7
    print("Theta: ", theta)

    J, theta = descenso_gradiente(X,y, theta, alpha, iteraciones)

    print("Theta (cerca de la óptima): ", theta)

    prec = prueba(X_prueba,y_prueba,theta)

    print("Precisión: ", prec)









         








    