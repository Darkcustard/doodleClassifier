import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras import metrics


limitVram = True

if limitVram:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)



dataDictionary = {

    'airplane' : np.array([1,0,0,0,0,0]),
    'bee' : np.array([0,1,0,0,0,0]),
    'banana' : np.array([0,0,1,0,0,0]),
    'eiffel' : np.array([0,0,0,1,0,0]),
    'bicycle' : np.array([0,0,0,0,1,0]),
    'bulldozer' : np.array([0,0,0,0,0,1]),

}

def loadDataset(path:str, labelName:str):

    x = np.load(path)
    length = x.shape[0]
    x = np.reshape(x,(length,28,28))
    print(x.shape)

    label = dataDictionary[labelName]    
    y = np.tile(label,(length,1))

    return (x, y)





def buildClassifier():

    #define model
    print('building model...')
    
    classifier = Sequential()

    #input
    classifier.add(Input(shape=(28,28,1)))

    #hidden
    classifier.add(Conv2D(32,(3,3),padding="same"))
    classifier.add(MaxPooling2D(pool_size=(2,2),padding="same"))
    classifier.add(Dropout(0.1))
    
    classifier.add(Conv2D(32,(3,3),padding="same"))
    classifier.add(MaxPooling2D(pool_size=(2,2),padding="same"))
    
    classifier.add(Flatten())
    
    classifier.add(Dense(72,activation="relu"))
    classifier.add(Dense(36,activation="relu"))

    #output
    classifier.add(Dense(6, activation="softmax"))
    
    
    #compile
    classifier.compile(loss = 'binary_crossentropy', run_eagerly=True ,optimizer = 'adam', metrics=[metrics.CategoricalAccuracy()])
    
    return classifier



def trainClassifier(classifier, epochs:int, batch_size:int, x_train:np.array, y_train:np.array):

    for x in range(epochs):

        print(f'Starting Epoch: {x+1}')
        classifier.fit(x_train,y_train)
        


    print("Saving model.")
    classifier.save('classifier.ai')


airplaneX, airplaneY = loadDataset('src/train/airplane.npy','airplane')
beeX, beeY = loadDataset('src/train/bee.npy','bee')
bananaX, bananaY = loadDataset('src/train/banana.npy','banana')
eiffelX, eiffelY = loadDataset('src/train/eiffel.npy','eiffel')
bicycleX, bicycleY = loadDataset('src/train/bicycle.npy','bicycle')
bulldozerX, bulldozerY = loadDataset('src/train/bulldozer.npy','bulldozer')

def spliceData():

    x = []
    y = []

    exhausted = False

    i = 0
    
    while not exhausted:

        if i < len(airplaneX):

            x.append(airplaneX[i])
            y.append(airplaneY[i])
            i+=1

        if i < len(beeX):

            x.append(beeX[i])
            y.append(beeY[i])
            i+=1            

        if i < len(bananaX):

            x.append(bananaX[i])
            y.append(bananaY[i])
            i+=1

        if i < len(bulldozerX):

            x.append(bulldozerX[i])
            y.append(bulldozerY[i])
            i+=1

        if i < len(eiffelX):

            x.append(eiffelX[i])
            y.append(eiffelY[i])
            i+=1


        if i < len(bicycleX):

            x.append(bicycleX[i])
            y.append(bicycleY[i])
            i+=1

        
        if i >= len(airplaneX) and i >= len(beeX) and i >= len(bicycleX) and i >= len(bulldozerX) and i >= len(eiffelX) and i >= len(bananaX):
            exhausted = True

    return (np.array(x),np.array(y))


x,y = spliceData()
classifier = buildClassifier()
trainClassifier(classifier,5,50,x,y)
