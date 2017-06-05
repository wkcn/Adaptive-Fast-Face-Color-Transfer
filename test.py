#coding=utf-8
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.misc import imshow, imresize

def RGB2YCbCr(rgb):
    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]
    Y = 0.257*R+0.504*G+0.098*B+16
    Cb = -0.148*R-0.291*G+0.439*B+128
    Cr = 0.439*R-0.368*G-0.071*B+128
    return np.dstack([Y, Cb, Cr])

def RGB2Lab(rgb):
    R = rgb[:,:,0] / 255.0
    G = rgb[:,:,1] / 255.0
    B = rgb[:,:,2] / 255.0
    T = 0.008856
    M, N = R.shape
    s = M * N
    RGB = np.r_[R.reshape((1, s)), G.reshape((1, s)), B.reshape((1, s))]
    MAT = np.array([[0.412453,0.357580,0.180423],
           [0.212671,0.715160,0.072169],
           [0.019334,0.119193,0.950227]])
    XYZ = np.dot(MAT, RGB)
    X = XYZ[0,:] / 0.950456
    Y = XYZ[1,:]
    Z = XYZ[2,:] / 1.088754


    XT = X > T
    YT = Y > T
    ZT = Z > T

    Y3 = np.power(Y, 1.0/3)
    fX = np.zeros(s)
    fY = np.zeros(s)
    fZ = np.zeros(s)

    fX[XT] = np.power(X[XT], 1.0 / 3)
    fX[~XT] = 7.787 * X[~XT] + 16.0 / 116

    fY[YT] = Y3[YT] 
    fY[~YT] = 7.787 * Y[~YT] + 16.0 / 116

    fZ[ZT] = np.power(Z[ZT], 1.0 / 3)
    fZ[~ZT] = 7.787 * Z[~ZT] + 16.0 / 116

    L = np.zeros(YT.shape)
    a = np.zeros(fX.shape)
    b = np.zeros(fY.shape)

    L[YT] = Y3[YT] * 116 - 16.0
    L[~YT] = 903.3 * Y[~YT] 

    a = 500 * (fX - fY)
    b = 200 * (fY - fZ)

    return np.dstack([L.reshape(R.shape), a.reshape(R.shape), b.reshape(R.shape)])

def Lab2RGB(Lab):
    M, N, C = Lab.shape
    s = M * N

    L = Lab[:,:,0].reshape((1, s)).astype(np.double)
    a = Lab[:,:,1].reshape((1, s)).astype(np.double)
    b = Lab[:,:,2].reshape((1, s)).astype(np.double)

    T1 = 0.008856
    T2 = 0.206893

    fY = np.power((L + 16.0) / 116, 3.0)
    YT = fY > T1
    fY[~YT] = L[~YT] / 903.3
    Y = fY.copy()

    fY[YT] = np.power(fY[YT], 1.0 / 3)
    fY[~YT] = 7.787 * fY[~YT] + 16.0 / 116

    fX = a / 500.0 + fY
    XT = fX > T2
    X = np.zeros((1, s)) 
    X[XT] = np.power(fX[XT], 3)
    X[~XT] = (fX[~XT] - 16.0 / 116) / 7.787

    fZ = fY - b / 200.0
    ZT = fZ > T2
    Z = np.zeros((1, s))
    Z[ZT] = np.power(fZ[ZT], 3)
    Z[~ZT] = (fZ[~ZT] - 16.0 / 116) / 7.787

    X = X * 0.950456
    Z = Z * 1.088754
    MAT = np.array([[ 3.240479,-1.537150,-0.498535],
       [-0.969256, 1.875992, 0.041556],
        [0.055648,-0.204043, 1.057311]])
    RGB = np.dot(MAT, np.r_[X,Y,Z])
    R = RGB[0, :].reshape((M,N))
    G = RGB[1, :].reshape((M,N))
    B = RGB[2, :].reshape((M,N))
    return np.clip(np.round(np.dstack([R,G,B]) * 255), 0, 255).astype(np.uint8)



def count(w):
    return dict(zip(*np.unique(w, return_counts = True)))
def count_array(w, size):
    d = count(w)
    return np.array([d.get(i, 0) for i in range(size)])


def deal(rgb):
    y = RGB2YCbCr(rgb)
    b = (y[:,:,1] >= 77) & (y[:,:,1] <= 127) & (y[:,:,2] >= 133) & (y[:,:,2] <= 173)
    lab = RGB2Lab(rgb)
    rrr = Lab2RGB(lab)
    Sa = count_array(lab[:,:,1][b], 256) 
    Sb = count_array(lab[:,:,2][b], 256) 


def face_color_transfer(source, target):
    pass

# RGB
source = mpimg.imread("pic/boy.jpg")
target = mpimg.imread("pic/black.jpg")
deal(source)

plt.subplot(121)
plt.title("source")
plt.imshow(source)
plt.subplot(122)
plt.title("target")
plt.imshow(target)
plt.show()

result = face_color_transfer(source, target)
plt.imshow(result)
plt.show()
