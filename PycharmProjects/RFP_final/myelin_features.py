import vigra
import numpy as np
from scipy import ndimage

def tensorEigenvalues(l, conn, rawFloat32):
    print "tensor EigenValues "
    tensorEigenvalues = vigra.filters.tensorTrace(rawFloat32)
    conn.send(tensorEigenvalues)
    conn.close()

def gaussianGradientMagnitude(conn, rawFloat32, sigma, RF_on_off):
    print "gaussian gradientMag..."
    gaussianGradientMagnitude = vigra.filters.gaussianGradientMagnitude(rawFloat32, sigma)
    if RF_on_off == 1:
        gaussianGradientMagnitude = np.ravel(gaussianGradientMagnitude.T)
        gaussianGradientMagnitude = gaussianGradientMagnitude.reshape(gaussianGradientMagnitude.shape[0], 1)
        conn.send(gaussianGradientMagnitude)
        conn.close()
    else:
        conn.send(gaussianGradientMagnitude)
        conn.close()

def laplacianOfGaussian(conn, rawFloat32, RF_on_off):
    print "laplplacian of Gaussian: "
    laplacianOfGaussian = vigra.filters.laplacianOfGaussian(rawFloat32)
    if RF_on_off == True:
        laplacianOfGaussian = np.ravel(laplacianOfGaussian.T)
        laplacianOfGaussian = laplacianOfGaussian.reshape(laplacianOfGaussian.shape[0], 1)
        conn.send(laplacianOfGaussian)
        conn.close()
    else:
        conn.send(laplacianOfGaussian)
        conn.close()


def hessOfGaussEV(conn, rawFloat32, sigma, RF_on_off):
    print "hessOfGaussEV with sigma = ", sigma
    hessGaussEV4D = vigra.filters.hessianOfGaussianEigenvalues(rawFloat32, sigma)
    # shape for RF 4D -> 3D matrices
    if RF_on_off == 1:
        hessGaussEVRF = np.zeros((LEN_DATA, 3), np.float32)
        for i in range(3):
            hessGaussEVRF[:,i] = np.ravel(hessGaussEV4D[:, :, :, i].T)
        conn.send(hessGaussEVRF)
        conn.close()
    else:
        conn.send(hessGaussEV4D)
        conn.close()

def hessOfGauss3D(conn, mag, rawFloat32, sigma, RF_on_off):
    print "hessOfGauss3D with sigma = ", sigma
    hessOfGauss3D = vigra.filters.hessianOfGaussian3D(rawFloat32, sigma)
    if RF_on_off == 1:
        if mag == 1:
            hessOfGauss3DRF = np.zeros((LEN_DATA, 6), np.float32)
        if mag == 8:
            hessOfGauss3DRF = np.zeros((LEN_DATA, 6), np.float32)
        for i in range(6):
            hessOfGauss3DRF[:, i] = np.ravel(hessOfGauss3D[:, :, :, i].T)
        conn.send(hessOfGauss3DRF)
        conn.close()
    else:
        conn.send(hessOfGauss3D)
        conn.close()

def discErosion(conn, rawUint8, radius, RF_on_off):
    print "discErosion with Radius: ", radius
    discErosion = vigra.filters.discErosion(rawUint8, radius)
    if RF_on_off == 1:
        discErosion = np.ravel(discErosion.T)
        discErosion = discErosion.reshape(discErosion.shape[0], 1)
    conn.send(discErosion)
    conn.close()




def connectedCompFeat(conn, firstBinClosing, secondDilatation, threshHold, max_size, RF_on_off, rawUint8):
    print "clearedImGauss.. "
    # Threshold
    rawBinary = np.where(rawUint8 > threshHold, 1, 0)
    # Remove small black particles
    binClosingVigra = vigra.filters.discClosing(rawBinary.astype(np.uint8), firstBinClosing)
    # change background
    tempColorRotate = (binClosingVigra.astype(np.uint8) -1)
    # remove connected Components smaller 90 000 pixels
    labels, count = ndimage.label(tempColorRotate.astype(np.bool))
    sizes = np.bincount(labels.ravel())
    mask_sizes = sizes > max_size
    mask_sizes[0]=0
    print "---------------"
    clearedImage = mask_sizes[labels]
    # regrow areas
    clearedImage = ndimage.binary_dilation(clearedImage, iterations=secondDilatation)
    # clearedImageEr = vigra.filters.discErosion(clearedImage.astype(np.uint8), 2)
    # expand areas smoothely
    clearedImageGauss = vigra.filters.gaussianSmoothing(clearedImage.astype(np.float32), 7)
    if RF_on_off == True:
        clearedImageGauss = np.ravel(clearedImageGauss.T)
        clearedImageGauss = clearedImageGauss.reshape(clearedImageGauss.shape[0], 1)
    print "cleared Image Gauss fct: ", clearedImageGauss.shape
    conn.send(clearedImageGauss)
    conn.close()

