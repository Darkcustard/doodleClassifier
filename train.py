import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

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

    model = Sequential()

    return model