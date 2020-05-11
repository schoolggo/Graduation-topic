"""
    Artificial Neural Network
    ANN.py
    Created on 2018/11/01
    @author garyvvallstar
    """
import numpy as np
import math
import pickle

class Neu:
    '''
        Artificial Neural Network
        '''
    
    def __init__(self):
        self.input_n = 3
        self.hidden_n = 3
        self.output_n = 1
        
        self.learn_speed = 0.01
        self.alpha = 0.5
        self.rms_flag = 999999
        self.epoch = 10
        self.people = 10
        
        self.wo_1 = np.random.random(size=(self.hidden_n, self.input_n))
        self.do_1 = np.random.random(size=(self.hidden_n, 1))
        self.wo_2 = np.random.random(size=(self.output_n, self.hidden_n))
        self.outp = []
        self.fq = []
        self.outq = []
        self.outr = []
        self.error_out = 0
    
    def saveneu(self, name):
        '''
            write neu to file
            '''
        f = open(name, 'wb')
        # dump the object to a file
        pickle.dump(self, f)
        f.close()
    
    def createnetwork(self):
        self.wo_1 = np.random.random(size=(self.hidden_n, self.input_n))
        self.do_1 = np.random.random(size=(self.hidden_n, 1))
        self.wo_2 = np.random.random(size=(self.output_n, self.hidden_n))
    
    def Activative_Function(slef, x):
        return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    
    def D_Activative_Function(self, x):
        return 4 / (np.exp(x) + np.exp(-x)) ** 2
    
    def OutPut_p(self, x):
        return np.array(np.mat(x).T, dtype='float64')
    
    def func_q(self, w, op, d):
        return np.array(np.dot(w, op) + d, dtype='float64')
    
    def OutPut_q(self, fq):
        return np.array(self.Activative_Function(fq), dtype='float64')
    
    def OutPut_r(self, w, oq):
        return np.array(np.dot(w, oq), dtype='float64')
    
    def bperror(self, Answer, result):
        return 0.5 * sum((Answer - result)) ** 2
    
    def change_wo_2(self, w2, E, oq):
        return np.array(w2 + self.learn_speed * np.dot(E, oq.T), dtype='float64')
    
    def change_do_1(self, d, E, w2, dfq):
        return np.array(d + self.learn_speed * dfq * np.dot(w2.T, E), dtype='float64')
    
    def change_wo_1(self, w1, E, w2, dfq, op):
        return np.array(w1 + self.learn_speed * np.dot(np.dot(w2.T,E) * dfq, op.T), dtype='float64')
    
    def forward(self, X_train_std):
        '''
            caluculate the result
            '''
        self.outp = self.OutPut_p(X_train_std)
        self.fq = self.func_q(self.wo_1, self.outp, self.do_1)
        self.outq = self.OutPut_q(self.fq)
        self.outr = self.OutPut_r(self.wo_2, self.outq)
        return self.outr
    
    def backward(self, y_train_std):
        '''
            adjust weight
            '''
        self.error_out = y_train_std - self.outr
        self.wo_2 = self.change_wo_2(self.wo_2, self.error_out, self.outq)
        self.do_1 = self.change_do_1(self.do_1, self.error_out, self.wo_2, np.array(self.D_Activative_Function(self.fq)))  # 此處outq 應該改為fq
        self.wo_1 = self.change_wo_1(self.wo_1, self.error_out, self.wo_2, np.array(self.D_Activative_Function(self.fq)), self.outp)

