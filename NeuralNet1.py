import numpy as np
from csvIO import*

#set number and label Train Data then shuffle 3 class
def labelTrainData(trainNum):
    classNum1=trainNum//3
    classNum2=trainNum//3
    classNum3=trainNum//3
    allreaddata=inputCSVtofloat("PCAtraindata.csv")
    allreaddata=np.transpose(allreaddata).tolist()

    label=[]
    label.extend([1]*1000)
    label.extend([2]*1000)
    label.extend([3]*1000)
    data=np.transpose(allreaddata+[label]).tolist()

    traindata=[]
    class1=np.array(data[0:classNum1])
    class2=np.array(data[1000:classNum2+1000])
    class3=np.array(data[2000:classNum3+2000])
    traindata.extend(class1)
    traindata.extend(class2)
    traindata.extend(class3)
    random.shuffle(traindata)
    random.shuffle(traindata)
    random.shuffle(traindata)
    A2DoutputCSV(traindata,"traindata.csv")


# train the neural network model with 1-hidden layer
def train(eta):
    

    traindata=inputCSVtofloat("traindata.csv")

    #layer1W=[[0.015,0.010,0.018,0.019,0.013,0.010],[0.010,0.014,0.016,0.013,0.017,0.011],[0.019,0.015,0.014,0.018,0.011,0.019]]
    #OutputW=[[0.016,0.010,0.019],[0.013,0.013,0.012],[0.011,0.014,0.018],[0.016,0.017,0.017],[0.015,0.013,0.011],[0.014,0.012,0.019],[0.017,0.015,0.013]]

    layer1W=np.random.rand(3,6)*0.01-0.05
    OutputW=np.random.rand(7,3)*0.01-0.05

    layer1WMtrx=np.matrix(layer1W)
    OutputWMtrx=np.matrix(OutputW)

    
    print(OutputWMtrx)
    print(layer1WMtrx)

    count=0

    for data in traindata+traindata:
        count+=1

        Class=int(data[2])
        target=[0.0,0.0,0.0]
        target[Class-1]=1.0
        #print("T\n",target)
        DesignMtrx=np.matrix([data[0],data[1],1.0])
        layer1A=np.dot(DesignMtrx,layer1WMtrx)
        #print("1A\n",layer1A)
        layer1Z=np.reciprocal( np.add( np.exp(np.negative(layer1A)),1 ))
        layer1Z=np.append(layer1Z,[[1]],axis=1)
        #print("1Z\n",layer1Z)

        OutputA=np.dot(np.matrix(layer1Z),OutputWMtrx)
        #print("OA\n",OutputA)
        OutputY=np.exp(OutputA)
        OutputSumRec=np.reciprocal( np.sum(OutputY) )
        OutputY=np.multiply(OutputY,OutputSumRec)
        #print("Y\n",OutputY)

        diff=np.add(OutputY,np.negative(target))
        #print("Odiff\n",diff)
        gradient_OutputW=np.dot( np.matrix(layer1Z.T),np.matrix(diff) )
        #print("gradOW\n",gradient_OutputW)

        OutputWMtrx=np.matrix(  np.add(OutputWMtrx,np.multiply(-eta,gradient_OutputW)) )

        if count%100==0:
            print("newOW\n",OutputWMtrx)

        expNeg_layer1A=np.exp(np.negative(layer1A))
        d_h=np.add(expNeg_layer1A,1.0)
        d_h=np.reciprocal( np.multiply(d_h,d_h) )
        d_h=np.multiply(expNeg_layer1A,d_h)

        #print("d_sigmoid\n",d_h)
        
        diff1=np.dot( np.matrix(diff),OutputWMtrx.T )
        #print("L1diff\n",diff1)
        diff1=np.multiply(d_h,np.delete(diff1,-1,1))

        gradient_layer1W=np.dot( DesignMtrx.T,np.matrix(diff1) )
        #print("grad1W\n",gradient_layer1W)
        layer1WMtrx=np.matrix( np.add(layer1WMtrx,np.multiply(-eta,gradient_layer1W)) )

        if count%100==0:
            print("newlayer1W\n",layer1WMtrx)
        
        

    A2DoutputCSV(layer1WMtrx.tolist(),"NN1layer1W.csv")
    A2DoutputCSV(OutputWMtrx.tolist(),"NN1OutputW.csv")    
        
# the neural network model with 1-hidden layer result
def NN1Output(num):
    classNum=num//3
    print("Model Output...")
    allreaddata=inputCSVtofloat("PCAtraindata.csv")
    vailddata=allreaddata[1000-classNum:1000]+allreaddata[2000-classNum:2000]+allreaddata[3000-classNum:3000]
    print(len(vailddata))
    layer1WMtrx=np.matrix( inputCSVtofloat("NN1layer1W.csv") )
    OutputWMtrx=np.matrix( inputCSVtofloat("NN1OutputW.csv") )

    modelOutput=[]
    for data in vailddata:
        DesignMtrx=np.matrix([data[0],data[1],1.0])
        layer1A=np.dot(DesignMtrx,layer1WMtrx)
        layer1Z=np.reciprocal( np.add( np.exp(np.negative(layer1A)),1 ))
        layer1Z=np.append(layer1Z,[[1]],axis=1)
        OutputA=np.dot(np.matrix(layer1Z),OutputWMtrx)
        OutputY=np.exp(OutputA)
        OutputSumRec=np.reciprocal( np.sum(OutputY) )
        OutputY=np.multiply(OutputY,OutputSumRec)
        modelOutput.extend( OutputY.flatten().tolist() )
        

    A2DoutputCSV(modelOutput,"NN1modelOutput.csv")  

#validate result probability
def NN1valid(fname):
    print("validate...")
    prob=inputCSVtofloat(fname)
    result=[]
    for index in range(len(prob)):
        if prob[index][0]>prob[index][1] and prob[index][0]>prob[index][2]:
            result.append(1)
        elif prob[index][1]>prob[index][0] and prob[index][1]>prob[index][2]:
            result.append(2)
        else:
            result.append(3)
    print(result)
    
    errorNum=0
    classnum=len(prob)//3
    for x in range(classnum):
        if result[x] != 1:
            errorNum=errorNum+1
        if result[x+classnum] != 2:
            errorNum=errorNum+1
        if result[x+classnum+classnum] != 3:
            errorNum=errorNum+1
    
    print("Error rate: %f"%(errorNum/len(prob)))