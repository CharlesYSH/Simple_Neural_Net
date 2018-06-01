from PIL import Image
import numpy as np
from csvIO import*

#read BMP image file
def readBMP(bmpfile):
    im = Image.open(bmpfile)
    im.load()
    imheight, imwidht = im.size
    #print("Read %d X %d image" %(imwidht,imheight))

    imgArray=[]
    for row in range(imheight):
        for col in range(imwidht):
            apixel = im.getpixel((row,col))
            imgArray.append(apixel)
    #print(imgArray)
    return imgArray

#Average pixel array
def pixelAvg(data):
    return np.average(data)

#Average a pixel postion from total photo
def photoAvg(imgNParray):
    TimgNParray=imgNParray.transpose()
    AvgPixelList=[]
    for pixelindex in range(900):
        avgPixel=pixelAvg(TimgNParray[pixelindex])
        AvgPixelList.append(avgPixel)
    return AvgPixelList
 
#create covariance matrix
def CovMtrx(bmpfile):
    imgArray=readBMP(bmpfile)
    imgAvg=pixelAvg(imgArray)

    imgMtrx=np.mat(imgArray,float)
    muMtrx=np.mat(np.full((1,900),-imgAvg))
    XMtrx=np.add(imgMtrx,muMtrx)
    covMtrx=np.dot(XMtrx.transpose(),XMtrx)
    covMtrx=np.dot(1/899,covMtrx)

    print("covMtrx OK!")
    eigenValue, eigenVector = np.linalg.eig(covMtrx)
    print(np.amax(eigenVector))
    ZMtrx=np.dot(XMtrx,eigenVector)
    print(ZMtrx.item(0,1))    
    #print(np.amax(ZMtrx))

#photo PCA to the data
def findEigenFace():
    imgArray=[]
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class1\\faceTrain1_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS1")
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class2\\faceTrain2_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS2")
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class3\\faceTrain3_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS3")

    
    photonumber=len(imgArray)
    print("Input %d photos."%photonumber)

    imgNParray=np.array(imgArray)
    avgPixelList=photoAvg(imgNParray)
    #print(avgPixelArray)
    #print(len(avgPixelArray))
    muArray = np.tile(np.negative(np.array(avgPixelList)), (len(imgArray),1) )
    muMtrx=np.mat(muArray)
    #print(muArray)
    #print(len(muArray))
    imgMtrx=np.mat(imgNParray)
    XMtrx=np.add(imgMtrx,muMtrx)
    
    covMtrx=np.dot(XMtrx.transpose(),XMtrx)
    covMtrx=np.dot(1/(photonumber-1),covMtrx)
    print("covMtrx OK!")
    print(covMtrx.shape)
    eigenValue, eigenVector = np.linalg.eig(covMtrx)
    print(eigenValue)

    print(np.amax(eigenValue))
    #A1DoutputCSV(eigenValue,"eigenValue.csv")
    print(eigenVector)
    W1W2Vector=[]
    print(eigenVector[:,0])
    coefw1=eigenVector[:,0]
    coefw2=eigenVector[:,1]
    for index in range(len(eigenVector)):
        W1W2Vector.append([coefw1.item(index,0).real,coefw2.item(index,0).real])

    A2DoutputCSV(W1W2Vector,"PCAeigenVector.csv")

    WMtrx=np.mat(W1W2Vector)
    reduceDimMtrx=np.dot(XMtrx,WMtrx)
    print(reduceDimMtrx)
    A2DoutputCSV(reduceDimMtrx.A,"traindata.csv")

#photo PCA to the data use tool
def toolfindEigenFace():
    from matplotlib.mlab import PCA
    imgArray=[]
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class1\\faceTrain1_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS1")
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class2\\faceTrain2_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS2")
    for filenum in range(1,1001):
        BMPfile = "Data_Train\\Class3\\faceTrain3_"+str(filenum)+".bmp"
        bmpArray=readBMP(BMPfile)
        imgArray.append(bmpArray)
    print("CLASS3")

    
    photonumber=len(imgArray)
    print("Input %d photos."%photonumber)

    imgNParray=np.array(imgArray)
    
    results = PCA(imgNParray)

    print(results.s)

    muArray = np.tile(np.negative(results.mu), (len(imgArray),1) )
    muMtrx=np.mat(muArray)
    imgMtrx=np.mat(imgNParray)
    XMtrx=np.add(imgMtrx,muMtrx)
    XMtrx=np.mat(XMtrx)

    W1W2Vector=[]
    coefw1=results.Wt[0]
    coefw2=results.Wt[1]
    W1W2Vector.append(coefw1)
    W1W2Vector.append(coefw2)
    W1W2Vector=np.transpose(W1W2Vector)
    print(W1W2Vector)

    A2DoutputCSV(W1W2Vector,"PCAeigenVector.csv")
    WMtrx=np.mat(W1W2Vector)
    reduceDimMtrx=np.dot(XMtrx,WMtrx)
    print(reduceDimMtrx)
    A2DoutputCSV(reduceDimMtrx.A,"PCAtraindata.csv")



