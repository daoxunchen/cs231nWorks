import cifar
import NearestNeighbor
import numpy as np

a,b,c,d=cifar.load_cifar10(1)

imgNum=1000

nn=NearestNeighbor.NearestNeighbor()
nn.train(a[:imgNum,:],b[:imgNum])
yy=nn.predict(c[:imgNum,:])

# print 'acc: %f' % (np.mean(yy==d[:imgNum]))
print 'acc: %d' % (np.sum(yy==d[:imgNum]))
