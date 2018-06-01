import numpy as np
import matplotlib.pyplot as plt
from csvIO import*
import bmpPCA
import NeuralNet1 as NN1
import NeuralNet2 as NN2


conductPCA=0
usetoolPCA=1
plotPCA=0
CreateTrainData=0
Train_1LayerModel=0
Valid_1LayerModel=0
Train_2LayerModel=0
Valid_2LayerModel=0
plotNN1=0
creatplotNN2data=0

if conductPCA:
    if usetoolPCA:
        bmpPCA.toolfindEigenFace()
    else:
        bmpPCA.findEigenFace()



if plotPCA:
    dataplot=inputCSVtofloat("PCAtraindata.csv")
    pc1=[]
    pc2=[]
    for x in dataplot:
        pc1.append(x[0])
        pc2.append(x[1])

    plt.plot(pc1[0:999], pc2[0:999], 'r.',pc1[1000:1999], pc2[1000:1999], 'g.',pc1[2000:2999], pc2[2000:2999], 'b.')
    plt.show()

if CreateTrainData:
    NN1.labelTrainData(2700)

if Train_1LayerModel:
    NN1.train(0.0037)

if Valid_1LayerModel:
    NN1.NN1Output(300)
    NN1.NN1valid("NN1modelOutput.csv")

if Train_2LayerModel:
    NN2.train(0.0009)

if Valid_2LayerModel:
    NN2.NN2Output(300)
    NN2.NN2valid("NN2modelOutput.csv")


if plotNN1:
    dataplot=inputCSVtofloat("PCAtraindata.csv")
    pc1=[]
    pc2=[]
    for x in dataplot:
        pc1.append(x[0])
        pc2.append(x[1])

    plt.plot(pc1[0:999], pc2[0:999], 'r.',pc1[1000:1999], pc2[1000:1999], 'g.',pc1[2000:2999], pc2[2000:2999], 'b.')
    plt.plot([-0.4179,-225.6], [-0.2504,-1000])
    plt.plot([-0.4179,1000], [-0.2504,-319.58])
    plt.plot([-0.4179,-634.29], [-0.2504,1000])
    plt.show()

if creatplotNN2data:
    NN2.NN2plot()