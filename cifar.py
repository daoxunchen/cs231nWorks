
def unpickle(file):
    import cPickle
    fo=open(file,'rb')
    dic=cPickle.load(fo)
    fo.close()
    return dic


def datToArray(dic):
    import numpy as np
    x=dic['data']
    y=dic['labels']
    return (x,y)

def load_cifar10(filesNum=5,dirname='/home/weiyu/cifar-10-batches-py/'):
    import numpy as np
    Xtr,Ytr=datToArray(unpickle(dirname+'data_batch_1'))
    for i in xrange(filesNum-1):
        d=unpickle(dirname+'data_batch_'+str(i+2))
        resX,resY=datToArray(d)
        Xtr=np.concatenate((Xtr,resX))
        Ytr=np.concatenate((Ytr,resY))

    Xte,Yte=datToArray(unpickle(dirname+'test_batch'))
    return (Xtr.astype(int),np.array(Ytr).astype(int),Xte.astype(int),np.array(Yte).astype(int))
