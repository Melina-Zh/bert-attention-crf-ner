#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:11:19 2019

@author: Melina
"""

def preprossor(filename,output):
    f=open(filename,'r')
    out=open(output,'w')
    flag=0
    for line in f:
        text=[]
        label=[]
        labels=[]
        split=line.split("####")
        label=split[1]
        labellist=label.split(' ')
        for i in labellist:
            tur=i.split('=')
            tur[0] = tur[0].lower()
            text.append(tur[0])
            if tur[1]=='T' and flag==0:
                labels.append("B-AP")
                flag=1
            elif tur[1]=='T' and flag==1:
                labels.append("I-AP")
            else:
                flag=0
                labels.append('O')
        out.write(' '.join(text)+"|||"+' '.join(labels)+'\n')
    out.close()
#preprossor("14semeval_rest_train.txt","14semeval_rest_train_set.txt")
#preprossor("14semeval_rest_test.txt","14semeval_rest_test_set.txt")
preprossor("14semeval_laptop_test.txt","14semeval_laptop_test_set.txt")
preprossor("14semeval_laptop_train.txt","14semeval_laptop_train_set.txt")
preprossor("16semeval_rest_test.txt","16semeval_rest_test_set.txt")
preprossor("16semeval_rest_train.txt","16semeval_test_train_set.txt")

