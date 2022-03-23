import numpy as np

def eval_numerical_gradient(f,x):
    fx = f(x) #original point의 함수값 구하기
    grad = np.zeros(x.shape)
    h = 0.00001

    #iterate over all indexes in x
    it = np.nditer(x, flags=['multi_index'],op_flags =['readwrite'])
    while not it.finished:
        
        #evaluate function at x+h
        ix = it.multi_index
        old_value = x[ix]
        x[ix] = old_value + h #increment by h
        fxh = f(x) #evaluate f(x+h)
        x[ix] = old_value #restore to previous value

        grad[ix] = (fxh-fx)/h
        it.iternext() #next dimension으로 가기

    return grad