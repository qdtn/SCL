__author__ = 'quynhdo'
from scipy import linalg
import numpy as np
# Implements Structural Corresponding Learning Algorithm
# Linear version


class SCL:
    def __init__(self):
        pass

    def predictor(self, X_train, Y_train):
        '''
        implement a single predictor
        :param X_train: input matrix
        :param Y_train: output vector
        :return:
        '''
        from sklearn.linear_model import SGDClassifier
        clf = SGDClassifier(loss="modified_huber", penalty="l2")
        clf.fit(X_train, Y_train)
        return clf.coef_


    def apply(self, X, pivot_feas, Y, h, X_labelled, masked_feas=None):
        '''
        X - train data used to predict pivot features (Come from the unsupervised data)
        Y - train labels used to predict pivot features
        pivot_feas - the vector indicate the pivot features in the feature vector
        h - dimension of the output
        X_labelled - train data for the main task
        masked_feas: in some case, we don't use all the non-pivot features for the prediction, just put it here
        '''
        mask = np.asarray(pivot_feas)
        rs = []
        for i in range(np.sum(pivot_feas)):
            Xi = None
            if masked_feas is not None:
                Xi = X * Y[i].T
            else:
                Xi = X * mask
            w = self.predictor(Xi, Y[i])

            rs.append(w.T)
        rs = np.asarray(rs)
        rs = rs.reshape(rs.shape[0],rs.shape[1])
        rs = rs.T
        u,s,v = linalg.svd(rs, full_matrices=True)

        return np.dot(X_labelled, u[0:h,:].T)




if __name__ == "__main__":

    '''
    test case
    "the girl"
    "the man"
    "girl man"
    feature = x, x[-1]
    bag of word "the girl man"
    pivot features: x = girl, x =man => indexes = 1,2
    '''
    X = np.asarray( [[1 , 0, 0, 0 , 0,  0], [0,1,0,1,0,0], [1,0 ,0, 0,0,0], [0,0,1,1,0,0],[0,1,0,0,0,0] , [0,0,1,0,1,0]] )
    pivots = [0,1,1,0,0,0]

    Y = [np.asarray([0, 1,0,0,1,0]),np.asarray([0, 0,0,1,0,1])]

    scl = SCL()
    print (scl.apply(X, pivots, Y, 2,X))

