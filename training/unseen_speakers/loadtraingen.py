# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 18:03:50 2017

@author: Alan
"""
import sys
import pickle

file=open('traingen.pkl','rb')
data=pickle.load(file)


a=(1,2)
print(sys.getsizeof(a))