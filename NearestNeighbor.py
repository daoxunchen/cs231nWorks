import numpy as np

class NearestNeighbor(object):
    def __init__(self):
        pass

    def L1distance(self,x):
        return np.sum(np.abs(self.xtr-x),axis=1)

    def L2distance(self,x):
        return np.sum(np.square(self.xtr-x),axis=1)
        # return np.sqrt(np.sum(np.square(self.xtr-x),axis=1))

    DistanceType={'L1':L1distance,'L2':L2distance}

    def train(self,x,y):
        self.xtr=x
        self.ytr=y

    def predict(self,x,dis='L1'):
        num_test=x.shape[0]
        ypred=np.zeros(num_test,dtype=self.ytr.dtype)
        process=0
        for i in xrange(num_test):
            if i%100==0:
                process+=1
                print process
            # distances=np.sum(np.abs(self.xtr-x[i,:]),axis=1)
            # distances=np.sqrt(np.sum(np.square(self.xtr-x[i,:]),axis=1))
            distances=self.DistanceType.get(dis)(self,x[i,:])
            # print distances
            min_index=np.argmin(distances)
            ypred[i]=self.ytr[min_index]

        return ypred

