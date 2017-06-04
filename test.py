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

def RGB2XYZ(rgb):
    var_R = rgb[:,:,0]
    var_G = rgb[:,:,1]
    var_B = rgb[:,:,2]
    X = var_R * 0.4124 + var_G * 0.3576 + var_B * 0.1805
    Y = var_R * 0.2126 + var_G * 0.7152 + var_B * 0.0722
    Z = var_R * 0.0193 + var_G * 0.1192 + var_B * 0.9505
    return np.dstack([X,Y,Z])

def XYZ2LAB(xyz):
    X = xyz[:,:,0]
    Y = xyz[:,:,1]
    Z = xyz[:,:,2]
    Xn = 95.047
    Yn = 100.0
    Zn = 108.883
    def f(k):
        e = (6.0 / 29) ** 3 
        b = (k > e)
        res = np.zeros(k.shape)
        res[b] = np.power(k[b], 1.0 / 3)
        res[~b] = 1.0 / 3 * ((29.0 / 6) ** 2) * k[~b] + 4.0 / 29
        return res 
    L = 116 * f(Y / Yn) - 16
    A = 500 * (f(X / Xn) - f(Y / Yn))
    B = 200 * (f(Y / Yn) - f(Z / Zn))

def deal(pic):
    y = RGB2YCbCr(pic)
    b = (y[:,:,1] >= 77) & (y[:,:,1] <= 127) & (y[:,:,2] >= 133) & (y[:,:,2] <= 173)
    plt.imshow(b, "gray")
    plt.show()

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
