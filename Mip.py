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

NUM_ITERATION = 20
NUM_CONTRAINTS = 2
RANDOM_LOWER_LIMIT = 0
RANDOM_UPPER_LIMIT = 200

LIMIT_CONST_LOWER = 500
LIMIT_CONST_UPPER = 10000

LIST_ANSWER =[]
LIST_QUESTION =[]

NUM_FEATURE = 5

MEAN_Y = []
STD_Y = []


def data_mining():
    for x in range(NUM_ITERATION):
        objectiveMatrix = randomList()
        constraintMatrixOne = randomList()
        constraintMatrixTwo  = randomList()

        answersMatrix = []

        constraintOneConstant = random.randrange(LIMIT_CONST_LOWER,LIMIT_CONST_UPPER)
        constraintTwoConstant = random.randrange(RANDOM_LOWER_LIMIT,LIMIT_CONST_UPPER)

        solution = solveLinearEquation(constraintMatrixOne,constraintOneConstant, constraintMatrixTwo,constraintTwoConstant, objectiveMatrix)


        if(solution[0] != 0 and solution[1]!= 0 and solution[0] != 100 and solution[1] != 100):

            #This is the place where the first matric is added
            answersMatrix.append(constraintMatrixOne)
            answersMatrix.append([constraintOneConstant, 0])

            #This is the place where the second matric is added
            answersMatrix.append(constraintMatrixTwo)
            answersMatrix.append([constraintTwoConstant, 0])

            #This is the place where the objective is

            answersMatrix.append(objectiveMatrix)

           # constraintMatrixTwo.append(constraintTwoConstant)

            LIST_ANSWER.append(solution)
            LIST_QUESTION.append(answersMatrix)

    print("This is the list of question: ")
    print(LIST_QUESTION)

    print("This is the list of answers: ")
    print(LIST_ANSWER)

    arr_questions = numpy.array(LIST_QUESTION)
    arr_questions = arr_questions.reshape((-1,NUM_FEATURE))

    arr_answers=numpy.array(LIST_ANSWER)

    #print((numpy.array(padded_lst).shape))



    #Save Question


   # np.save('nn_y.npy', numpy.array(LIST_ANSWER))

    np.save ('x_listVersion.npy' , zscore(arr_questions))
    np.save ('y_listVersion.npy', zscore(arr_answers))

  #  MEAN_Y = [numpy.mean(numpy.array(LIST_ANSWER)[:,col]) for col in range(2) ]
   # STD_Y = [numpy.std(numpy.array(LIST_ANSWER)[:,col]) for col in range(2) ]

    #np.save ('mean.npy' , MEAN_Y)
    #np.save ('std.npy' , STD_Y)

    print("This is the list of the array of questions: ")
    print("constraintMatrixOne constraintOneConstant constraintMatrixTwo constraintTwoConstant objectiveMatrix")
    print(arr_questions)

    print("This is the list of the array of answers")
    print(arr_answers)

    print("Total answers: ")
    print(len(arr_answers))

    print()
    print()

    print("---- NORMALIZED VALUES OF THE DATA AFTER Z_SCORE NORMALIZATION -----")
    print ("constraintMatrixOne constraintOneConstant constraintMatrixTwo constraintTwoConstant objectiveMatrix")
    print (zscore(arr_questions))
    print (zscore(arr_answers))
    print("This is the length of the arrray: ")
    print(zscore(arr_questions).shape)

    print("---- MEAN AND STD VALUES FOR THE Y -----")
    print(MEAN_Y)
    print(STD_Y)




def randomList():
    a = random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT)
    b= random.randrange(RANDOM_LOWER_LIMIT,RANDOM_UPPER_LIMIT)
    return [a,b]


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




def RNN(x,y, mean, std ):
    tensorflow.compat.v1.reset_default_graph ()

    model = Sequential()
    model.add (Dense (128 , activation='relu' , input_shape=(None, NUM_FEATURE)))
    model.add (Dense (64 , activation='relu'))
    model.add (Dense (32 , activation='relu'))
    model.add (Dense (16 , activation='relu'))
    model.add (Dense (8 , activation='relu'))
    model.add (Dense (4 , activation='relu'))
    model.add (Dense (2 , activation='softmax'))

    #sgd=keras.optimizers.Adamax (lr=0.15)
    model.compile (loss='mse' , optimizer='adamax' , metrics=[ 'accuracy' ] , )
    plot_model (model , to_file='model.png' , show_shapes=True , show_layer_names=True)
    #vis_utils.plot_model (model , to_file='model.png' , show_shapes=True , show_layer_names=True)

    history = model.fit (x ,y , epochs=500 , batch_size=32 )

    # Plot the training and validation loss over epochs
    plt.plot (history.history [ 'loss' ] , label='Training loss')
   # plt.plot (history.history [ 'Validation loss' ] , label='Validation loss')
    plt.xlabel ('Epoch')
    plt.ylabel ('Loss')
    plt.legend ()
    plt.show ()

    # Assuming you have already trained your model and have the test data
    y_pred = model.predict(x)
    print(y_pred.shape)
    print(y.shape)

    print("R2")

    r2 = r2_score(y,y_pred)
   # f1score = f1_score(y[0][0], y_pred[0][0] , average='macro')

    print(r2)


    print("----")
  #  ans = model.predict(normalize(numpy.array([[2],[1],[20], [2], [3] ,[50] ,[2] ,[1]]).T))

    ans = model.predict(np.reshape(x[0],(-1,1)).T)
    print("Answer CNN--- ",ans)
    print("Real answer normalized --- " , y[0] )
    print()

    val1 = (ans[ 0 ][ 0 ] * std[0]) + mean[0]
    val2 = (ans[ 0 ][ 1 ] * std[1]) + mean[1]
    print("Denormalized values" , val1, val2)

    val3=(y [ 0 ] [ 0 ] * std [ 0 ]) + mean [ 0 ]
    val4=(y [ 0 ] [ 1 ] * std [ 1 ]) + mean [ 1 ]
    print("Denormalized values from normalized y ---" ,val3,val4)
    acan = np.load('nn_y.npy')
    print("Acutla calculate ans " , acan)



def main():
    data_mining()
    name = numpy.array([0,1, 2,3,4])
    x = numpy.load('x_listVersion.npy')
    numpy.insert(x,0,name)
    y = numpy.load('y_listVersion.npy')
    print(y[:20])
    print(x[:10])
    name = [5 , 6]
    numpy.insert(y,0,name)

    print(x[2])

    print(numpy.shape(x),numpy.shape(y))
    #sns.pairplot (data=pandas.DataFrame(numpy.concatenate((x,y), axis=1)) , x_vars = [0,1,2,3,4], y_vars=[5,6])
   #plt.show()

    MEAN_Y = numpy.load ('mean.npy')
    STD_Y =numpy.load ('std.npy')

    print(len(x))
    print("Meand and Stds")
    print(MEAN_Y[0],MEAN_Y[1])
    print(STD_Y[0],STD_Y[1])
    RNN(x,y, MEAN_Y , STD_Y)
    print()
    print(solveLinearEquation([2,1],20 ,[3,3], 50, [2,1]))


main()