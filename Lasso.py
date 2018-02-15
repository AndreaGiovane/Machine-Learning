import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def loadDataset():
    X = np.genfromtxt('dataset.csv', delimiter=',')
    X = X[1:,:]
    mask = np.logical_or(X[:,0]==3, X[:,0]==5)
    X = X[mask, :]
    Y = X[:,0]
    X = X[:,1:]
    return X, Y

def fixDataset(X, Y):
    mask1 = Y == 3
    mask2 = Y == 5
    #create training
    X1 = X[mask1, :]
    Xm1 = X1[0:1000, :]
    Y1 = np.ones(1000)
    X2 = X[mask2,:]
    Xm2 = X2[0:1000, :]
    Y2 = np.ones(1000)*-1
    Xtr = np.concatenate((Xm1, Xm2))
    Ytr = np.concatenate((Y1, Y2))
    #create test
    nTe1 = len(X1[1000:,:])
    nTe2 = len(X2[1000:,:])
    Xte = np.concatenate((X1[1000:,:], X2[1000:,:]))
    Yte = np.concatenate((np.ones(nTe1), np.ones(nTe2)*-1))
    return Xtr, Ytr, Xte, Yte

def soft_treshold(x, lamb):
    if lamb > abs(x):
        return 0
    if x > 0:
        return x - lamb
    else:
        return x + lamb

def lasso(X, Y, lamb, iter):
    lamb = len(X) * lamb
    w = np.zeros(X.shape[1])
    precRes = np.ones(len(Y))*np.inf
    actualRes = Y.ravel().copy() #initialize the residue to the best solution
    delta = np.inf
    for i in range(iter):
        if i %100 == 0:
            print(i, end=' ')
        for j in range(len(w)):
            aux = X[:,j] * w[j] #extract the j variable and multiply it with the corresponding coefficient
            actualRes += aux #subtract it from the residue (you must add it to do this, due to how the residue is calculated
            if np.sum(X[:,j]) != 0:
                w[j] = soft_treshold(np.dot(X[:,j], actualRes), lamb)/np.sum(X[:,j]*X[:,j]) #compute the new j component
            aux =  X[:,j] * w[j]
            actualRes -= aux #add it to the residue (you must subtract it to do this, due to how the residue is calculated)
        delta = np.sum(np.abs(actualRes - precRes))
        if delta <  1e-4: #convergence
            print('\nConvergence obtained in {} iterations'.format(i))
            break
        precRes = actualRes.copy() #update the residue and repeat everything for the next component
    else:
        print('Reached the maximum number of iterations equal to {}.\nCurrent delta between residues equal to {}'.format(delta))
    return w

def calcErr(Ypred, Y):
    return np.mean(Ypred != Y)

def holdoutCV(X, Y, perc, rep, intRegPar):
    maxIter = 10000
    nRegPar = len(intRegPar)
    n = len(X)
    ntr = int(n*perc)
    Tm = np.zeros(nRegPar)
    Ts = np.zeros(nRegPar)
    Vm = np.zeros(nRegPar)
    Vs = np.zeros(nRegPar)
    for i, lamb in enumerate(intRegPar):
        print('try for ', lamb)
        repErrTr = np.zeros(rep)
        repErrTe = np.zeros(rep)
        for r in range(rep):
            print('\trip: ', r)
            mask = np.random.permutation(n)
            Xtr = X[mask[:ntr]]
            Xte = X[mask[ntr:]]
            Ytr = Y[mask[:ntr]]
            Yte = Y[mask[ntr:]]
            w = lasso(Xtr, Ytr, lamb, maxIter)
            repErrTr[r] = calcErr(np.sign(np.dot(Xtr, w)), Ytr)
            repErrTe[r] = calcErr(np.sign(np.dot(Xte, w)), Yte)
        Tm[i] = np.mean(repErrTr)
        Vm[i] = np.mean(repErrTe)
        Ts[i] = np.std(repErrTr)
        Vs[i] = np.std(repErrTe)
    imin = np.argmin(Vm)
    w = lasso(Xtr, Ytr, intRegPar[imin], maxIter)
    return intRegPar[imin], w, Tm, Ts, Vm, Vs

def main():
    intRegPar = np.logspace(-3, 0, 5)
    #intRegPar = np.linspace(1, 6, 6)

    perc = 0.7
    rep = 5

    X, Y = loadDataset()
    Xtr, Ytr, Xte, Yte = fixDataset(X, Y)

    l, w, Tm, Ts, Vm, Vs = holdoutCV(Xtr, Ytr, perc, rep, intRegPar)

    print('best lambda: ', l, '\nw: ', w, '\nTm: ', Tm, '\nTs: ', Ts, '\nVm: ', Vm, '\nVs: ', Vs)

    print('number of selected features', np.sum(w != 0))

    testError = calcErr(np.sign(np.dot(Xte, w)), Yte)
    print('\nError on test set built with the found w: ', testError)

    #first graph
    train_patch  = mpatches.Patch(color=(1,1,0), label='mean training error')
    test_patch   = mpatches.Patch(color=(0,1,1), label='mean validation error')
    lambda_patch = mpatches.Patch(color=(1,0,0), label='best lambda')

    plt.figure(figsize=(12,7))
    plt.loglog(intRegPar,Vm, c = (0,1,1))
    plt.fill_between(intRegPar, Vm - Vs, Vm+Vs)
    plt.loglog(intRegPar,Tm, c = (1,1,0))
    plt.fill_between(intRegPar, Tm - Ts, Tm+Ts)
    plt.plot([l,l], [1e-3,1], '--', c = (1,0,0))
    plt.ylim([0.5e-2, 2e-1])
    plt.xlim([1e-3, 1])
    plt.legend(handles=[train_patch, test_patch, lambda_patch])
    plt.xlabel('lambda')
    plt.ylabel('error')
    plt.title('holdout [trainP = {}, rep = {}]'.format(perc, rep))
    plt.show()
    plt.close()

    #second graph
    w1 = w[np.argsort(w)]
    w1 = w1[w1 != 0]

    plt.figure(figsize=(12,7))
    plt.bar(np.arange(len(w1)), w1)
    plt.ylim([-0.003, 0.003])
    plt.ylabel('magnitude')
    plt.xlabel('features')
    plt.show()
    plt.close()
    print(len(w1)/len(w))

    #third graph
    mask = np.zeros((28,28,3))
    w1 = w.reshape(28,28)
    for i in range(28):
        for j in range(28):
            if w1[i,j] > 0:
                mask[i,j] = [0,w1[i,j]*500,0]
            elif w1[i,j] < 0:
                mask[i,j] = [-w1[i,j]*500,0,0]


    train_patch  = mpatches.Patch(color=(1,0,0), label='3 label')
    test_patch   = mpatches.Patch(color=(0,1,0), label='5 label')
    lambda_patch = mpatches.Patch(color=(0,0,0), label='removed feature')

    plt.figure(figsize=(7,7))
    plt.imshow(mask)
    plt.legend(handles=[train_patch, test_patch, lambda_patch], mode="expand",bbox_to_anchor=(0., 1.05, 1., .102))
    plt.show()
    plt.close()

main()
