import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model , vis_utils
from mip import *
import  random

from mip import Model, xsum, maximize, BINARY
import numpy


import sklearn
from sklearn.linear_model import LinearRegression

NUM_ITERATION = 150
NUM_CONTRAINTS = 1
RANDOM_LOWER_LIMIT = 0
RANDOM_UPPER_LIMIT = 200

LIMIT_CONST_LOWER = 5000
LIMIT_CONST_UPPER = 100000

LIST_ANSWER =[]
LIST_QUESTION =[]

NUM_FEATURE = 5



def main():
    for x in range(NUM_ITERATION):
        objectiveMatrix = randomList()
        answer = sum(objectiveMatrix)
        constraintMatrixOne = randomList()

        constraintOneConstant = random.randrange(LIMIT_CONST_LOWER,LIMIT_CONST_UPPER)

        LIST_QUESTION.append(constraintMatrixOne)
        LIST_QUESTION.append(constraintOneConstant)
        LIST_QUESTION.append(objectiveMatrix)

        solution = solveLinearEquation(objectiveMatrix, constraintMatrixOne, constraintOneConstant , constraintOneConstant, 0 )


        LIST_ANSWER.append(solution)

    #print( (numpy.array(LIST_QUESTION).shape))

    #CNN(numpy.array(LIST_QUESTION),numpy.array(LIST_ANSWER),len(numpy.array(LIST_ANSWER)))

    linear_regression(numpy.array(LIST_QUESTION),numpy.array(LIST_ANSWER))







def randomList():
    a = random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT, )
    b= random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT)
    return [a,b]



def solveLinearEquation(objectiveMatrix, constraintMatrixOne, constraintMatrixTwo, constraintOneConstant,constraintTwoConstant):



    m = Model("LP")

    variableList = [m.add_var(var_type=INTEGER, lb = 0, ub = 100) for i in range(NUM_CONTRAINTS)]

    m.objective = maximize(xsum(objectiveMatrix[i] * variableList[i] for i in range(NUM_CONTRAINTS)))

    m += xsum(constraintMatrixOne[i] * variableList[i] for i in range(NUM_CONTRAINTS)) <= constraintOneConstant
    #m += xsum(constraintMatrixTwo[i] * variableList[i] for i in range(NUM_CONTRAINTS)) <= constraintTwoConstant

    m.optimize()

    selected = [variableList[i].x for i in range(NUM_CONTRAINTS) ]

    return selected

def CNN(x,y , col):
    tensorflow.compat.v1.reset_default_graph ()

    model=keras.Sequential (
        [
            keras.layers.Dense (48 , input_shape=(NUM_FEATURE ,)) ,
            keras.layers.Dense (24 , activation="relu") ,
            keras.layers.Dense (12 , activation="softmax") ,
            keras.layers.Dense (1, activation="softmax")

        ]
    )

    model.compile (loss='MSE' , optimizer='adam' , metrics=[ 'accuracy' ])
    plot_model (model , to_file='model.png' , show_shapes=True , show_layer_names=True)
    #vis_utils.plot_model (model , to_file='model.png' , show_shapes=True , show_layer_names=True)

    model.fit (x , y , epochs=500 , batch_size=32 )


def linear_regression(x,y):
    reg = LinearRegression()
    reg.fit(x,y)
    print("Result of linear regression" , reg.predict(numpy.array([[2],[1],[20] ,[1] ,[2]])))
    #print (c ([ 2 , 1 ] , 20 [ 2 , 3 ] , 50 , [ 1 , 2 ]))



main()