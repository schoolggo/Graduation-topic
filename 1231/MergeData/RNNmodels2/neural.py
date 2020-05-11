"""
Hammerstein Recurrent Neural Network
neural.py
"""

import random
import math
import pickle

class Neu:
    """
    Hammerstein-Wiener Recurrent Neural Network
    """

    def __init__(self):

        self.inputnumber = 5
        self.hidnumber = 2
        self.outputnumber = 1

        self.learnspeedwd = 0.006
        self.learnspeedab = 0.0006

        self.maxd = 0
        self.mind = 0

        self.X = [] # clear every time
        self.op = [] # clear every time

        self.w = [] # keep
        self.d1 = [] # keep
        self.fq = [] # clear every time
        self.oq = [] # clear every time

        self.BB = [] # keep
        self.AA = [] # keep
        self.fr = [] # clear every time
        self.orr = [] # keep

        self.CC = [] # keep
        self.fs = [] # clear every time
        self.os = [] # clear every time

        self.Y = [] # clear every time

        self.X_temp = [] # clear every time
        self.BB_temp = [] # clear every time
        self.CC_temp = [] # clear every time
        self.data_s = [] # clear every time
        self.data_r = [] # clear every time
        self.data_q = [] # clear every time
        self.error = [] # keep
        self.OD_or_a_plus = [] # keep
        self.OD_or_b_plus = [] # keep

        self.createnetwork()

    def saveneu(self, name):
        """
        write neu to file
        """
        f = open(name, 'wb')
        # dump the object to a file
        pickle.dump(self, f)
        f.close()

    def createzero(self, createlist, row, col):
        """
        create zero list
        """
        createlist = []
        for i in range(row):
            temp = []
            for j in range(col):
                temp.append(0)
            createlist.append(temp)
        return createlist

    def createrandom(self, createlist, row, col):
        """
        create random list
        """
        createlist = []
        for i in range(row):
            temp = []
            for j in range(col):
                temp2 = random.uniform(-0.9, 0.9)
                temp.append(temp2)
            createlist.append(temp)
        return createlist

    def createnetwork(self):
        """
        create the network structure
        """
        self.AA = self.createrandom(self.AA, self.hidnumber, self.hidnumber)
        self.BB = self.createrandom(self.BB, self.hidnumber, 1)
        self.CC = self.createrandom(self.CC, self.outputnumber, self.hidnumber)
        self.w = self.createrandom(self.w, self.hidnumber, self.inputnumber)
        self.d1 = self.createrandom(self.d1, self.hidnumber, 1)
        self.X = self.createzero(self.X, self.hidnumber, 1)
        self.Y = self.createzero(self.Y, self.outputnumber, 0)
        self.op = self.createzero(self.op, self.inputnumber, 1)
        self.fq = self.createzero(self.fq, self.hidnumber, 1)
        self.oq = self.createzero(self.oq, self.hidnumber, 1)
        self.fr = self.createzero(self.fr, self.hidnumber, 1)
        self.orr = self.createzero(self.orr, self.hidnumber, 1)
        self.fs = self.createzero(self.fs, self.outputnumber, 1)
        self.os = self.createzero(self.os, self.outputnumber, 1)
        self.X_temp = self.createzero(self.X_temp, self.hidnumber, 1)
        self.BB_temp = self.createzero(self.BB_temp, self.hidnumber, 1)
        self.CC_temp = self.createzero(self.CC_temp, self.outputnumber, self.hidnumber)
        self.data_s = self.createzero(self.data_s, self.outputnumber, 1)
        self.data_r = self.createzero(self.data_r, self.hidnumber, 1)
        self.data_q = self.createzero(self.data_q, self.hidnumber, 1)
        self.error = self.createzero(self.error, self.outputnumber, 1)
        self.OD_or_a_plus = self.createzero(self.OD_or_a_plus, self.hidnumber, self.hidnumber)
        self.OD_or_b_plus = self.createzero(self.OD_or_b_plus, self.hidnumber, 1)

    def cleartemporaltrain(self):
        """
        clear temp data every time
        """
        self.X_temp = self.createzero(self.X_temp, self.hidnumber, 1)
        self.BB_temp = self.createzero(self.BB_temp, self.hidnumber, 1)
        self.CC_temp = self.createzero(self.CC_temp, self.outputnumber, self.hidnumber)
        self.data_s = self.createzero(self.data_s, self.outputnumber, 1)
        self.data_r = self.createzero(self.data_r, self.hidnumber, 1)
        self.data_q = self.createzero(self.data_q, self.hidnumber, 1)
        self.X = self.createzero(self.X, self.hidnumber, 1)
        self.op = self.createzero(self.op, self.inputnumber, 1)
        self.fq = self.createzero(self.fq, self.hidnumber, 1)
        self.oq = self.createzero(self.oq, self.hidnumber, 1)

        self.fr = self.createzero(self.fr, self.hidnumber, 1)
        self.fs = self.createzero(self.fs, self.outputnumber, 1)
        self.os = self.createzero(self.os, self.outputnumber, 1)
        self.Y = self.createzero(self.Y, self.outputnumber, 0)

    def cleartemporalepoch(self):
        """
        clear temp data every epoch
        """
        self.orr = self.createzero(self.orr, self.hidnumber, 1)

        self.OD_or_a_plus = self.createzero(self.OD_or_a_plus, self.hidnumber, self.hidnumber)
        self.OD_or_b_plus = self.createzero(self.OD_or_b_plus, self.hidnumber, 1)

    def forward(self, inputlist):
        """
        calculate result
        """
        self.cleartemporaltrain()

        # layer 1: input layer
        for i in range(self.inputnumber):
            self.op[i][0] = inputlist[i]
            # print('op: ', self.op[i][0])

        # layer 2: hidden layer
        for i in range(self.hidnumber):
            for j in range(self.inputnumber):
                self.fq[i][0] = self.fq[i][0] + self.w[i][j] * self.op[j][0]
                # print('w: ', self.w[i][j])
            self.fq[i][0] = self.fq[i][0] + self.d1[i][0]
            # print('d1: ', self.d1[i][0])
            # print('fq: ', self.fq[i][0])

        for i in range(self.hidnumber):
            upper = math.exp(self.fq[i][0]) - math.exp(-self.fq[i][0])
            down = math.exp(self.fq[i][0]) + math.exp(-self.fq[i][0])
            self.oq[i][0] = upper / down
            # print('oq: ', self.oq[i][0])

        # layer 3: dynamic layer
        for i in range(self.hidnumber):
            self.X_temp[i][0] = self.orr[i][0]

        temp = []
        for i in range(self.hidnumber):
            temp.append(self.BB[i][0] * self.oq[i][0])
            # print('temp: ', temp[i])

        for i in range(self.hidnumber):
            for j in range(self.hidnumber):
                self.fr[i][0] = self.fr[i][0] + self.AA[i][j] * self.orr[j][0]
            self.fr[i][0] = self.fr[i][0] + temp[i]
            # print('fr: ', self.fr[i][0])

        for i in range(self.hidnumber):
            self.orr[i][0] = self.fr[i][0]

        # layer 4: output layer
        for i in range(self.outputnumber):
            for j in range(self.hidnumber):
                self.fs[i][0] = self.fs[i][0] + self.CC[i][j] * self.orr[j][0]
            # print('fs: ',self.fs[i][0])

        for i in range(self.outputnumber):
            self.os[i][0] = self.fs[i][0]

        for i in range(self.outputnumber):
            self.Y[i] = self.os[i][0]

        return self.Y

    def backward(self, expect):
        """
        adjust weight
        """
        # decide error
        for i in range(self.outputnumber):
            self.error[i][0] = expect[i] - self.Y[i]
            # print('error: ', self.error)

        for i in range(self.outputnumber):
            self.data_s[i][0] = self.error[i][0]
            # print('data_s: ', self.data_s[i][0])

        # update CC
        for i in range(self.outputnumber):
            for j in range(self.hidnumber):
                self.CC_temp[i][j] = self.CC[i][j]

        for i in range(self.outputnumber):
            for j in range(self.hidnumber):
                self.CC[i][j] = self.CC[i][j] + self.learnspeedab * self.data_s[i][0] * self.orr[j][0]
            # print('CC: ', self.CC[i])

        for i in range(self.outputnumber):
            for j in range(self.hidnumber):
                self.data_r[j][0] = self.data_s[i][0] * self.CC_temp[i][j]
                # print('data_r: ', self.data_r[j][0])

        # update BB
        for i in range(self.hidnumber):
            self.BB_temp[i][0] = self.BB[i][0]

        for i in range(self.hidnumber):
            self.OD_or_b_plus[i][0] = self.oq[i][0] + self.AA[i][i] * self.OD_or_b_plus[i][0]
            self.BB[i][0] = self.BB[i][0] + self.learnspeedab * self.data_r[i][0] * self.OD_or_b_plus[i][0]
            # print('BB: ', self.BB[i][0])
            # print('OD_or_b_plus: ', self.OD_or_b_plus[i][0])

        # update AA
        for i in range(self.hidnumber):
            for j in range(self.hidnumber):
                self.OD_or_a_plus[i][j] = self.X_temp[j][0] + self.AA[i][i] * self.OD_or_a_plus[i][j]
                self.AA[i][j] = self.AA[i][j] + self.learnspeedab * self.data_r[i][0] * self.OD_or_a_plus[i][j]
                # print('AA: ', self.AA[i][j])
                # print('OD_or_a_plus: ', self.OD_or_a_plus[i][j])

        for i in range(self.hidnumber):
            cos = math.exp(self.fq[i][0]) + math.exp(-self.fq[i][0])
            self.data_q[i][0] = self.data_r[i][0] * self.BB_temp[i][0] * 4 / (cos * cos)
            # print('data_q: ', self.data_q[i][0])

        # update d1
        for i in range(self.hidnumber):
            self.d1[i][0] = self.d1[i][0] + self.learnspeedwd * self.data_q[i][0]
            # print('d1: ', self.d1[i][0])

        # update w
        for i in range(self.hidnumber):
            for j in range(self.inputnumber):
                self.w[i][j] = self.w[i][j] + self.learnspeedwd * self.data_q[i][0] * self.op[j][0]
                # print('w: ', self.w[i][0])
                #　print('data_q: ', self.data_q[i][0])
                # print('op: ', self.op[j][0])
                # print('All: ', self.learnspeedwd * self.data_q[i][0] * self.op[j][0])
