import csv

#Output list-of-list to csv file
def A2DoutputCSV(OutputArray,outputname):

    with open(outputname, 'w', newline='') as csvfile:
        csvWrite = csv.writer(csvfile, dialect='excel')
        for rindex in range(0,len(OutputArray)):
            output=[]
            for index in range(0,len(OutputArray[0])):
                output.append( str(OutputArray[rindex][index]))
            csvWrite.writerow(output)

#Output list to csv file
def A1DoutputCSV(OutputArray,outputname):

    with open(outputname, 'w', newline='') as csvfile:
        csvWrite = csv.writer(csvfile, dialect='excel')
        for rindex in range(0,len(OutputArray)):
            output=[str(OutputArray[rindex])]
            csvWrite.writerow(output)

#Input csv file and retrun float list-of-list 
def inputCSVtofloat(inputname):

    with open(inputname, 'r', newline='') as csvfile:
        inputArray=[]
        for row in csv.reader(csvfile, dialect='excel'):
            rowArray=[]
            for x in row:
                rowArray.append(float(x))
            inputArray.append(rowArray)
        return inputArray
