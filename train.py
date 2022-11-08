import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, Input
from tensorflow.keras import metrics

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


airplaneX, airplaneY = loadDataset('src/train/airplane.npy','airplane')
beeX, beeY = loadDataset('src/train/bee.npy','bee')
bananaX, bananaY = loadDataset('src/train/banana.npy','banana')
eiffelX, eiffelY = loadDataset('src/train/eiffel.npy','eiffel')
bicycleX, bicycleY = loadDataset('src/train/bicycle.npy','bicycle')
bulldozerX, bulldozerY = loadDataset('src/train/bulldozer.npy','bulldozer')


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



def trainClassifier(classifier, epochs:int, batch_size:int):

    for x in range(epochs):

        print(f'Starting Epoch: {x+1}')
        classifier.fit(airplaneX,airplaneY,epochs=1,batch_size=batch_size)
        classifier.fit(beeX,beeY,epochs=1,batch_size=batch_size)
        classifier.fit(eiffelX,eiffelY,epochs=1,batch_size=batch_size)
        classifier.fit(bulldozerX,bulldozerY,epochs=1,batch_size=batch_size)
        classifier.fit(bananaX,bananaY,epochs=1,batch_size=batch_size)
        classifier.fit(bicycleX,bicycleY,epochs=1,batch_size=batch_size)

    print("Saving model.")
    classifier.save('classifier.ai')

classifier = buildClassifier()
trainClassifier(classifier,1,256)
