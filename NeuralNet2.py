import numpy as np
from csvIO import*

def FuncRectified(array): #if x<0 y=0 ; if x>0 y=x 
    value=np.multiply( np.add(array,np.abs(array)),0.5 )
    return value

def FuncD_Rectified(array): #if x<0 y=0 ; if x>0 y=1 
    value=np.multiply( np.add(array,np.abs(array)),0.5 )
    value=np.multiply(value,np.reciprocal(array))
    return value

def FuncSigmoid(array): #e^x/(1+e^x) #1/(1+e^-x)
    value=np.exp(array)
    value=np.multiply( value,np.reciprocal(np.add(value,1.0)) )
    #value=np.reciprocal( np.add( np.exp(np.negative(array)),1.0 ) )
    return value

def FuncSoftmax(array): #e^x/sum(e^x)
    Output=np.exp(array)
    recipSum=np.reciprocal( np.sum(Output) )
    Output=np.multiply(Output,recipSum)
    return Output

def FuncSoftplus (array): #ln(1+e^x)
    value=np.log( np.add(np.exp(array),1.0) )
    return value

# train the neural network model with 2-hidden layer
def train(eta):
    print("train NeuralNet 2 layer model ...")
    

    layer1W=[[0.017,0.018,-0.010,0.013,-0.016,0.011],[-0.014,0.015,0.018,-0.019,0.014,-0.013],[-0.011,0.017,-0.016,0.012,-0.015,0.018]]
    layer2W=[[0.015,0.010,0.018,0.019,0.013,0.010],[0.010,0.014,0.016,0.013,0.017,0.011],[0.019,0.015,0.014,0.018,0.011,0.019],[0.013,0.014,0.012,0.011,0.011,0.010],[0.019,0.016,0.013,0.017,0.015,0.014],[0.011,0.013,0.017,0.019,0.018,0.012],[0.015,0.011,0.014,0.016,0.012,0.013]]
    OutputW=[[0.016,0.010,0.019],[0.013,0.013,0.012],[0.011,0.014,0.018],[0.016,0.017,0.017],[0.015,0.013,0.011],[0.014,0.012,0.019],[0.017,0.015,0.013]]

    #layer1W=np.random.rand(3,6)*0.01-0.05
    #layer1W=np.random.rand(7,6)*0.01-0.05
    #OutputW=np.random.rand(7,3)*0.01-0.05

    traindata=inputCSVtofloat("traindata.csv")

    layer1WMtrx=np.matrix(layer1W)
    layer2WMtrx=np.matrix(layer2W)
    OutputWMtrx=np.matrix(OutputW)

    print("layerOutW\n",OutputWMtrx)
    print("layer2W\n",layer2WMtrx)
    print("layer1W\n",layer1WMtrx)

    for data in traindata:
        Class=int(data[2])
        target=[0.0,0.0,0.0]
        target[Class-1]=1.0
        #print("T\n",target)

        DesignMtrx=np.matrix([data[0],data[1],1.0])
        layer1A=np.dot(DesignMtrx,layer1WMtrx)
        #print("1A\n",layer1A)
        layer1Z=FuncRectified(layer1A)
        #print("1Z\n",layer1Z)
        layer1ZMtrx=np.matrix(np.append(layer1Z,[[1]],axis=1))

        layer2A=np.dot(layer1ZMtrx,layer2WMtrx)
        #print("2A\n",layer2A)
        layer2Z=FuncRectified(layer2A)
        #print("2Z\n",layer2Z)
        layer2ZMtrx=np.matrix(np.append(layer2Z,[[1]],axis=1))

        OutputA=np.dot(layer2ZMtrx,OutputWMtrx)
        #print("OA\n",OutputA)
        OutputY=FuncSoftmax(OutputA)
        #print("Y\n",OutputY)


        diffO=np.add(OutputY,np.negative(target))
        #print("Odiff\n",diffO)
        gradient_OutputW=np.dot( layer2ZMtrx.T,np.matrix(diffO) )
        #print("gradOW\n",gradient_OutputW)
        OutputWMtrx=np.matrix(  np.add(OutputWMtrx,np.multiply(-eta,gradient_OutputW)) )
        #print("newOW\n",OutputWMtrx)

        diff2=np.dot( np.matrix(diffO),OutputWMtrx.T )
        #
        d_h2=FuncD_Rectified(layer2A)
        #print("d_rectified\n",d_h2)
        diff2=np.multiply(d_h2,np.delete(diff2,-1,1))
        #print("L2diff\n",diff2)
        gradient_layer2W=np.dot( layer1ZMtrx.T,np.matrix(diff2) )
        #print("grad2W\n",gradient_layer2W)
        layer2WMtrx=np.matrix( np.add(layer2WMtrx,np.multiply(-eta,gradient_layer2W)) )
        #print("newlayer2W\n",layer2WMtrx)

        diff1=np.dot( np.matrix(diff2),layer2WMtrx.T )
        #
        d_h1=FuncD_Rectified(layer1A)
        #print("d_rectified\n",d_h1)
        diff1=np.multiply(d_h1,np.delete(diff1,-1,1))
        #print("L1diff\n",diff1)
        gradient_layer1W=np.dot( DesignMtrx.T,np.matrix(diff1) )
        #print("grad1W\n",gradient_layer1W)
        layer1WMtrx=np.matrix( np.add(layer1WMtrx,np.multiply(-eta,gradient_layer1W)) )
        #print("newlayer1W\n",layer1WMtrx)
        
    A2DoutputCSV(layer1WMtrx.tolist(),"NN2layer1W.csv")
    A2DoutputCSV(layer2WMtrx.tolist(),"NN2layer2W.csv")
    A2DoutputCSV(OutputWMtrx.tolist(),"NN2OutputW.csv")   
        

# the neural network model with 2-hidden layer result
def NN2Output(num):
    classNum=num//3
    print("Model Output...")
    allreaddata=inputCSVtofloat("PCAtraindata.csv")
    vailddata=allreaddata[1000-classNum:1000]+allreaddata[2000-classNum:2000]+allreaddata[3000-classNum:3000]
    print(len(vailddata))
    layer1WMtrx=np.matrix( inputCSVtofloat("NN2layer1W.csv") )
    layer2WMtrx=np.matrix( inputCSVtofloat("NN2layer2W.csv") )
    OutputWMtrx=np.matrix( inputCSVtofloat("NN2OutputW.csv") )

    modelOutput=[]
    for data in vailddata:
        DesignMtrx=np.matrix([data[0],data[1],1.0])

        layer1A=np.dot(DesignMtrx,layer1WMtrx)
        layer1Z=FuncRectified(layer1A)
        layer1ZMtrx=np.matrix(np.append(layer1Z,[[1]],axis=1))

        layer2A=np.dot(layer1ZMtrx,layer2WMtrx)
        layer2Z=FuncRectified(layer2A)
        layer2ZMtrx=np.matrix(np.append(layer2Z,[[1]],axis=1))

        OutputA=np.dot(layer2ZMtrx,OutputWMtrx)
        OutputY=FuncSoftmax(OutputA)

        modelOutput.extend( OutputY.flatten().tolist() )

    A2DoutputCSV(modelOutput,"NN2modelOutput.csv")  

#validate result probability
def NN2valid(fname):
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

def NN2plot():
    print("Model Output...")
    layer1WMtrx=np.matrix( inputCSVtofloat("NN2layer1W.csv") )
    layer2WMtrx=np.matrix( inputCSVtofloat("NN2layer2W.csv") )
    OutputWMtrx=np.matrix( inputCSVtofloat("NN2OutputW.csv") )

    modelOutput=[]
    for y in range(-500,700):
        result=[]
        for x in range(-1200,800):
            DesignMtrx=np.matrix([x,y,1.0])

            layer1A=np.dot(DesignMtrx,layer1WMtrx)
            layer1Z=FuncRectified(layer1A)
            layer1ZMtrx=np.matrix(np.append(layer1Z,[[1]],axis=1))

            layer2A=np.dot(layer1ZMtrx,layer2WMtrx)
            layer2Z=FuncRectified(layer2A)
            layer2ZMtrx=np.matrix(np.append(layer2Z,[[1]],axis=1))

            OutputA=np.dot(layer2ZMtrx,OutputWMtrx)
            OutputY=FuncSoftmax(OutputA)

            if OutputY.item((0,0))>OutputY.item((0,1)) and OutputY.item((0,0))>OutputY.item((0,2)):
                result.append(1)
            elif OutputY.item((0,1))>OutputY.item((0,0)) and OutputY.item((0,1))>OutputY.item((0,2)):
                result.append(2)
            else:
                result.append(3)
        print(y)
        modelOutput.append( result )

    A2DoutputCSV(modelOutput,"NN2plotOutput.csv")  