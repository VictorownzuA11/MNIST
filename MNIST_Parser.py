#python 3.7
import time, struct as st, numpy as np, pickle as pk

# python files
from helper import *

def main():
    net = network([784,16,10])
    images = 20
    labels_l = getLabels('train-labels.idx1-ubyte')
    images_l = getImages('train-images.idx3-ubyte',images)
    
    for i in range(0,images):
        net.update[0] += sigmoid(np.dot(net.weights[0],images_l[i]) + net.bias[0])
        for j in range(1,net.numLayers-1):
            net.update[j] += sigmoid(np.dot(net.weights[j],net.weights[j+1]) + net.bias[j])
    
    
class network(object):
    def __init__(self, neurons):
        self.numLayers = len(neurons)
        self.biases = [np.random.randn(x,1) for x in neurons[1:]]
        self.weights = [np.random.randn(x,y) for x, y in zip(neurons[:-1],neurons[1:])]
        self.update = [np.zeros((x,1)) for x in neurons[1:]]

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))
    
def dsigmoid(x):
    # d/dx (1 + e^(-x))^-1
    # -1(1 + e^(-x))^-2 * (-e^(-x))
    # e^(-x) / (1 + e^(-x))^2
    # e^(-x) * sigmoid(x)^2
    return np.exp(-x) * sigmoid(x)**2
    
def percepitron(value,bias):
    if value > bias:
        return 1
    return 0

def getLabels(filename):
    labels_l = []
    print('Getting labels')
    with open(filename,'rb') as f:
        chunk = f.read(8)
        MSB = st.unpack('>I',chunk[0:4])[0]
        labels = st.unpack('>I',chunk[4:8])[0]
        chunk = f.read(labels)
        for i in range(0,labels):
            labels_l.append(st.unpack('B',chunk[0+i:1+i])[0])
    print('Done getting labels')
    return labels_l

    
def getImages(filename,images_n):
    print('Getting images')
    with open(filename,'rb') as f:
        chunk = f.read(16)
        MSB = st.unpack('>I',chunk[0:4])[0]
        images = st.unpack('>I',chunk[4:8])[0]
        xDim = st.unpack('>I',chunk[8:12])[0]
        yDim = st.unpack('>I',chunk[12:16])[0]
        imageSize = xDim * yDim
        images_l = np.zeros((images,imageSize))
        count = 0
        for i in range(0,images_n):
            chunk = f.read(imageSize)
            printNum(chunk)
            for j in range(0,imageSize):
                images_l[i,j] = st.unpack('B',chunk[j:(j+1)])[0]
    print('Done getting images')
    return images_l

def layer(weights,bias,image,numNeurons):
    neurons = np.zeros((numNeurons,1))
    for i in range(0,numNeurons):
        neurons[i] = sigmoid(np.dot(weights[i],image)/numNeurons + bias[i])
    return neurons
    
def right(i,lables):
    if i == lables:
        return 1
    else:
        return 0
    
def backPropagation():
    cat = 1
    
if __name__ == '__main__':
    main()
    input("\nDone...press ENTER to exit")
