__author__ = 'lukas'
import vigra


def get_rawCube():
    # spaeter mit neuem DU
    "test"
    rawUint8 = vigra.impex.readHDF5('/home/lukas/Downloads/RAWtrainingMAG1.h5', 'raw')
    return rawUint8

def choseMyelinCube(i):
    print i
    return [12,12,12]
