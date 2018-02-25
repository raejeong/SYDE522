#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt 

def imshow(img):
    plt.imshow(img[:,:,0],cmap='gray')
    plt.show() 

def main():

    X, Y = np.expand_dims(np.load('X.npy'), 3), np.load('Y.npy')
    X, Y = X[100:,:,:,:], Y[100:]
    Xs, Ys = [], []
    for i in range(20):
        idx = np.where(Y==i)
        Xs.append(X[idx])
        Ys.append(Y[idx])
    
    newXs, newYs = [], []

    for i in range(20):
        newY = Ys[i][0]
        for j in range(4000):
            newX = np.zeros([200,200]).reshape((200,200,1))
            for kx in range(4):
                for ky in range(4):
                    idx = np.random.randint(Xs[i].shape[0])
                    currentX = Xs[i][idx,:,:,:]
                    h, w = currentX.shape[:2]
                    new_h, new_w = 50, 50
                    top = np.random.randint(0, h - new_h)
                    left = np.random.randint(0, w - new_w)
                    croppedX = currentX[top: top + new_h, left: left + new_w]
                    newX[kx*50:(kx+1)*50,ky*50:(ky+1)*50,:] = croppedX
            newXs.append(newX)
            newYs.append(newY)
    newXs = np.array(newXs).astype('uint8')
    newYs = np.array(newYs).astype('int64')
    np.save("moreX.npy", newXs)
    np.save("moreY.npy", newYs)
    print("{} images generated!".format(newYs.shape[0]))

if __name__ == "__main__":
    main()
