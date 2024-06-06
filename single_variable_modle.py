import keras
import tensorflow
from keras.models import Sequential
from keras.layers import Dense , Conv2D
from keras.utils import plot_model , vis_utils
from mip import *
import  random
from scipy.stats import zscore
import matplotlib.pyplot as plt
import seaborn as sns


from mip import Model, xsum, maximize, BINARY
import numpy
import pandas


import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

from sklearn.preprocessing import normalize

NUM_ITERATION = 50
NUM_CONTRAINTS = 2
RANDOM_LOWER_LIMIT = 0
RANDOM_UPPER_LIMIT = 200

LIMIT_CONST_LOWER = 500
LIMIT_CONST_UPPER = 10000

LIST_ANSWER =[]
LIST_QUESTION =[]

NUM_FEATURE = 8

MEAN_Y = []
STD_Y = []


#(constraintMatrixOne,constraintOneConstant, constraintMatrixTwo,constraintTwoConstant, objectiveMatrix)
def solveLinearEquation(constraintMatrixOne,constraintOneConstant, constraintMatrixTwo,constraintTwoConstant, objectiveMatrix):
    m = Model("LP")

    variableList = [m.add_var(var_type=INTEGER, lb = 0, ub = 1000) for i in range(NUM_CONTRAINTS)]

    m.objective = maximize(xsum(objectiveMatrix[i] * variableList[i] for i in range(NUM_CONTRAINTS)))

    m += xsum(constraintMatrixOne[i] * variableList[i] for i in range(NUM_CONTRAINTS)) <= constraintOneConstant
    m += xsum(constraintMatrixTwo[i] * variableList[i] for i in range(NUM_CONTRAINTS)) <= constraintTwoConstant
    m.optimize()
    selected = [variableList[i].x for i in range(NUM_CONTRAINTS) ]
    return selected


def z_normalize(arr,size):
    mean = [numpy.mean(arr[:,col]) for col in range(size)]
    std =  [numpy.std(arr[:,col]) for col in range(size)]
    arr = (arr - mean) / std
    return arr


def printStuffs():

    print("----- THESE ARE THE ARRAYS THAT WE HAVE FORMED---")
    print ("constraintMatrixOne constraintOneConstant constraintMatrixTwo constraintTwoConstant objectiveMatrix")
    print (numpy.array (LIST_QUESTION))
    print()
    print("This is the list of array of answers: ")
    print (numpy.array (LIST_ANSWER))
    print()
    print("This is the length of of the answers: ")
    print (len (numpy.array (LIST_ANSWER)))

    print ()
    print ()

    print ("---- NORMALIZED VALUES OF THE DATA AFTER Z_SCORE NORMALIZATION -----")
    print ("constraintMatrixOne constraintOneConstant constraintMatrixTwo constraintTwoConstant objectiveMatrix")
    print (zscore (numpy.array (LIST_QUESTION)))
    print()
    print("This is the list of array of answers: ")
    print (zscore (numpy.array (LIST_ANSWER)))
    print()
    print("This is the length of of the answers: ")
    print (len (numpy.array (LIST_ANSWER)))

    print ("---- MEAN AND STD VALUES FOR THE Y -----")
    print (MEAN_Y)
    print (STD_Y)

    # its the same as above so no point oops !
    print ("--- My normalization")
    print (z_normalize(numpy.array (LIST_ANSWER) , len (LIST_ANSWER [ 0 ])))



def randomList():
    a = random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT, )
    b= random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT)
    return [a,b]


def data_mining():


    for x in range(NUM_ITERATION):
        objectiveMatrix=randomList ()
        constraintMatrixOne=randomList ()
        constraintMatrixTwo  = randomList()

        answersMatrix = []

        constraintOneConstant = random.randrange(LIMIT_CONST_LOWER,LIMIT_CONST_UPPER)
        constraintTwoConstant = random.randrange(RANDOM_LOWER_LIMIT,LIMIT_CONST_UPPER)

        solution = solveLinearEquation(constraintMatrixOne,constraintOneConstant, constraintMatrixTwo,constraintTwoConstant, objectiveMatrix)

        if(solution[0] != 0 and solution[1]!= 0 and solution[0] != 100 and solution[1] != 100):

            #This is the place where the first matric is added
            answersMatrix.extend(constraintMatrixOne)
            answersMatrix.append(constraintOneConstant)

            #This is the place where the second matric is added
            answersMatrix.extend(constraintMatrixTwo)
            answersMatrix.append(constraintTwoConstant)

            #This is the place where the objective is

            answersMatrix.extend(objectiveMatrix)

            #This is where the answes are appended
            LIST_ANSWER.append(solution)

            #Final data is added here for quesitons
            LIST_QUESTION.append(answersMatrix)

    #Save Question

    np.save ('x_experiment.npy' , zscore(numpy.array(LIST_QUESTION)))
    np.save ('y_experiment.npy', zscore(numpy.array(LIST_ANSWER)))

    MEAN_Y = [numpy.mean(numpy.array(LIST_ANSWER)[:,col]) for col in range(2) ]
    STD_Y = [numpy.std(numpy.array(LIST_ANSWER)[:,col]) for col in range(2) ]

    np.save ('mean.npy' , MEAN_Y)
    np.save ('std.npy' , STD_Y)
    printStuffs()



def RNN(x,y, mean, std ):
    tensorflow.compat.v1.reset_default_graph ()

    model = Sequential()
    model.add (Dense (128 , activation='gelu' , input_shape=(None, NUM_FEATURE)))
    model.add (Dense (64 , activation='elu'))
    model.add (Dense (32 , activation='gelu'))
    model.add (Dense (16 , activation='elu'))
    model.add (Dense (8 , activation='gelu'))
    model.add (Dense (4 , activation='elu'))
    model.add (Dense (2 , activation='softmax'))

    model.compile (loss='mse' , optimizer='adam' , metrics=[ 'accuracy' ] , )
    plot_model (model , to_file='model.png' , show_shapes=True , show_layer_names=True)
    history = model.fit (x ,y , epochs=250 , batch_size=32)

    # Plot the training and validation loss over epochs
    plt.plot (history.history [ 'loss' ] , label='Training loss')
    plt.xlabel ('Epoch')
    plt.ylabel ('Loss')
    plt.legend ()
    plt.show ()

    # Assuming you have already trained your model and have the test data
    y_pred = model.predict(x)


    print()
    print("THIS IS THE Y PREDICTION SHAPE AND ACTUAL SHAPE OF THE Y DATA SET ")
    print(y_pred.shape)
    print(y.shape)

    print()
    print("-----R2------")
    r2 = r2_score(y,y_pred)
    print(r2)
    print("----")


    ans = model.predict(np.reshape(x[0],(-1,1)).T)
    print("Answer BY THE MODEL FOR THE FIRST DATA --- ",ans)
    print("Real answer normalized --- " , y[0] )
    print()

    val1 = (ans[ 0 ][ 0 ] * std[0]) + mean[0]
    val2 = (ans[ 0 ][ 1 ] * std[1]) + mean[1]
    print("Denormalized values that we recieve from the model: " , val1, val2)


    val3=(y [ 0 ] [ 0 ] * std [ 0 ]) + mean [ 0 ]
    val4=(y [ 0 ] [ 1 ] * std [ 1 ]) + mean [ 1 ]
    print("Denormalized values we recieve from the actual answers: " ,val3,val4)

def linear_regression(x,y ):
    reg = LinearRegression()
    reg.fit(x,y)
    print("Result of linear regression" , (reg.predict(numpy.array(np.reshape(x[0],(-1,1)).T))))
    print(y[0])
    #print (c ([ 2 , 1 ] , 20 [ 2 , 3 ] , 50 , [ 1 , 2 ]))


def main():
    data_mining()
    #Doing some lable work to plot some variables
    name = numpy.array([0,1, 2,3,4])
    x = numpy.load('x.npy')
    numpy.insert(x,0,name)
    y = numpy.load('y.npy')
    name = [5 , 6]
    numpy.insert(y,0,name)
    sns.pairplot (data=pandas.DataFrame (numpy.concatenate ((x , y) , axis=1)) , x_vars=[ 0 , 1 , 2 , 3 , 4 ] ,
              y_vars=[ 5 , 6 ])
    plt.show ()

    print(numpy.shape(x),numpy.shape(y))


    MEAN_Y = numpy.load ('mean.npy')
    STD_Y =numpy.load ('std.npy')

    print(len(x))

   #We print mean here
    print("Mean and Stds: ")
    print(MEAN_Y[0],MEAN_Y[1])
    print(STD_Y[0],STD_Y[1])
    print()

    RNN(x,y, MEAN_Y , STD_Y)



# def solveLinearEquation(constraintMatrixOne , constraintOneConstant , constraintMatrixTwo , constraintTwoConstant ,  objectiveMatrix) :
    #([ [ 2 ] , [ 1 ] , [ 20 ] , [ 2 ] , [ 3 ] , [ 50 ] , [ 2 ] , [ 1 ] ])
    print(solveLinearEquation([2,1],20 ,[3,3], 50, [2,1]))

    #print(x[0])


main()