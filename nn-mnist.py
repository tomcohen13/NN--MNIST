from scipy.io import loadmat
from random import sample
import numpy as np
from matplotlib import pyplot as plt

def nn(X,Y,test):

    # Compute Euclidean distances using some linear algebra.

    T2 = np.matrix(np.square(test).sum(axis = 1)) # that's all the ti^2
    
    X2 = np.matrix(np.square(X).sum(axis = 1)) # that's all the xj^2

    TXT = np.dot(test, X.T)

    #now we can construct a table of all distances using broadcasting:
    euclideans = np.sqrt(T2.T - 2*TXT + X2)
    
    #find argmin {di1, ... din} for each row (that is, for each test example ti)
    indices = euclideans.argmin(axis = 1)

    preds = np.array([Y[j][0][0] for j in indices]).astype('int')
    return preds

         

if __name__ == '__main__':
    ocr = loadmat('ocr.mat')
    num_trials = 10
    mean_errors, sd_errors = [], []
    
    for n in [1000, 2000, 4000, 8000]:
        test_err = np.zeros(num_trials)
        for trial in range(num_trials):
            sel = sample(range(len(ocr['data'].astype('float'))),n)
            preds = nn(ocr['data'].astype('float')[sel], ocr['labels'][sel], ocr['testdata'].astype('float'))
            test_err[trial] = np.mean(preds != ocr['testlabels'])
        print("%d\t%g\t%g" % (n,np.mean(test_err), np.std(test_err)))
        
        mean_errors.append(np.mean(test_err))
        sd_errors.append(np.std(test_err))

    
    plt.errorbar(np.array([1000,2000,4000,8000]), mean_errors, marker = 'o', c = 'b', yerr = sd_errors)
    plt.axis([0, 8500, min(mean_errors) - 0.01, 0.01 + max(mean_errors)])
    plt.xlabel('n (sample size)')
    plt.ylabel('error rate')
    plt.show()
        

    
