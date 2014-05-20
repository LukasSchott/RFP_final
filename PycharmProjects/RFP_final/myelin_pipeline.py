__author__ = 'lukas'

import numpy as np
from multiprocessing import Process, Pipe
from myelin_features import *
from get_rawCube import *
from sklearn.ensemble import RandomForestClassifier

""" classifier for 256+(32*2), 256+(32*2), 128+(32*2) ->
320x320x192
Total Cube 10.000x10.000x5x000
-> 60.000 classifications -> 15 000 -> 1250 pro Prozessor

TO
see methods
gaussDivergence
master HCI
load RF_classifier
"""

# still needs to be implemented


# 1=on and writes, 0=view only
RF_on_off = 0
# magnitude attention hessGaussEVRF has to be changed
# mag8 31744, mag1 16711680=256*256*255
mag = 1
# hessOfGaussEV_on_off 1:on, 0: off
sighOG1 = 12
# hessOfGaussEV_on_off 1:on, 0: off
sighOG2  = 16
# hessOfGauss3D_on_off 1:on, 0: off
hessOfGauss3D_on_off = 0
# discErosion_on_off 1: on, 0: off
radiusDE = 1
# laplacianOfGaussian_on_off
laplacianOfGaussian_on_off = 0
# gaussianGradientMagnitude
sigGM = 1
# sigmaGaussianDivergence = 0.5-2 optimal
sigmaGD = 1

# connComp
connectedCompFeat_on_off = 0
threshHold1 = 160
firstBinClosing1 = 6
secondDilatation1 = 3
thirdErosion1 = 2
max_size1 = 5000

connectedCompFeat_on_off2 = 1
threshHold2 = 160
firstBinClosing2 = 6
secondDilatation2 = 3
thirdErosion2 = 2
max_size2 = 50000


# tensorEigenvalues // KLAPPT NOCH NICHT
# does not work yet
tensorEigenvalues_on_off = 0

# testOrTraining = 'testarea' or = 'training' or 'berlinCube' or 'juliasCube'
testOrTraining = 'training'
# not needed anymore

# still in need:
CubeItterator = 1
Offset = choseMyelinCube(CubeItterator)
rawUint8 = get_rawCube()
LEN_DATA = np.ravel(rawUint8)
rawFloat32 = vigra.gaussianSmoothing(rawUint8.astype(np.uint8), 0.6)


# creating the pipes for the features
parent_hessOfGaussEV12, child_hessOfGaussEV12 = Pipe()
parent_hessOfGaussEV16, child_hessOfGaussEV16 = Pipe()
parent_gaussianGradientMagnitude, child_gaussianGradientMagnitude = Pipe()
parent_discErosion, child_discErosion = Pipe()
parent_connectedCompStrong, child_connectedCompStrong = Pipe()
parent_connectedComWeak, child_connectedCompWeak = Pipe()

numberOfProcesses = 2

p = range(numberOfProcesses)

# p[0] = Process(target=loadRF)


p[1] = Process(target=hessOfGaussEV,
             args=(child_hessOfGaussEV12,
                1, rawFloat32, sighOG1, RF_on_off, LEN_DATA))

# p[2] = Process(target=hessOfGaussEV,
#              args=(child_hessOfGaussEV16,
#                    1, rawFloat32, sighOG2, RF_on_off))
#
# p[3] = Process(target=gaussianGradientMagnitude,
#              args=(child_gaussianGradientMagnitude, rawFloat32, sigGM, RF_on_off))
#
# p[4] = Process(target=discErosion,
#              args=(child_discErosion, rawUint8, radiusDE, RF_on_off))
#
# p[5] = Process(target=connectedCompFeat,
#              args=(child_connectedCompStrong, firstBinClosing1, secondDilatation1,
#                    threshHold1, max_size1, RF_on_off, rawUint8))
#
# p[6] = Process(target=connectedCompFeat,
#              args=(child_connectedCompWeak, firstBinClosing2, secondDilatation2,
#                    threshHold2, max_size2, RF_on_off, rawUint8))


for i in range(numberOfProcesses):
    p[i].start()

print "all processes decleared"

# recive data from the processes
parent_hessOfGaussEV12 = parent_hessOfGaussEV12.recv()
# hessOfGaussEV16 = parent_hessOfGaussEV16.recv()
# gaussianGradientMagnitude = parent_gaussianGradientMagnitude.recv()
# discErosion = parent_discErosion.recv()
# connectedCompStrong = parent_connectedCompStrong.recv()
# connectedComWeak = parent_connectedComWeak.recv()

for i in range(numberOfProcesses):
    p[i].join()

testData = concatenate

clf_Btest = RandomForestClassifier(n_estimators=10, n_jobs=2)

clf_Btest.fit(testOrTraining)
