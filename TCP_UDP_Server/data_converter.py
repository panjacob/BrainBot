import numpy as np
import communication_parameters

def bytesToStruct(rawData):
    data_struct = np.zeros((communication_parameters.samples, communication_parameters.channels))

    #32 bit unsigned words reorder
    rawDataArray = np.array(rawData)
    rawDataArray = rawDataArray.reshape((communication_parameters.words, 3))
    rawDataArray = np.transpose(rawDataArray)
    normaldata = rawDataArray[2,:]*(256**3) + rawDataArray[1,:]*(256**2) + rawDataArray[0,:]*256 + 0
    for i in range(communication_parameters.samples):
        for j in range(communication_parameters.channels):
            data_struct[i, j] = normaldata[i + j].astype('int32') 
    return data_struct