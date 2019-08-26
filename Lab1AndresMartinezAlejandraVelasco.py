# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 20:29:03 2019

@author: Andrés
"""

#LAB 1: DECISION TREES
#DD2421 MACHINE LEARNING
# ANDRÉS MARTÍNEZ MORA & ALEJANDRA VELASCO SÁNCHEZ-VILLARES

# ENTROPY CALCULATION
import monkdata as m
from dtree import entropy,averageGain,select,mostCommon,allPruned

ent1=entropy(m.monk1)
print("The entropy of MONK-1 is {}".format(ent1))
ent2=entropy(m.monk2)
print("The entropy of MONK-2 is {}".format(ent2))
ent3=entropy(m.monk3)
print("The entropy of MONK-3 is {}".format(ent3))


# INFORMATION GAIN CALCULATION
# FOR MONK-1 ATTRIBUTES

info_gain1=[0]*6
for i in range(6):
    info_gain1[i]=averageGain(m.monk1,m.attributes[i])
    print("Information gain in MONK-1 tree for a{} is {}".format(i+1,info_gain1[i]))
    
    
# FOR MONK-2 ATTRIBUTES

info_gain2=[0]*6
for i in range(6):
    info_gain2[i]=averageGain(m.monk2,m.attributes[i])
    print("Information gain in MONK-2 tree for a{} is {}".format(i+1,info_gain2[i]))
    
    

# FOR MONK-3 ATTRIBUTES

info_gain3=[0]*6
for i in range(6):
    info_gain3[i]=averageGain(m.monk3,m.attributes[i])
    print("Information gain in MONK-3 tree for a{} is {}".format(i+1,info_gain3[i]))
    

# DATA SPLITTING

# ATTRIBUTE A5 IN MONK-1
# A5 HAS VALUES {1,2,3,4}
    
monk1_1=select(m.monk1,m.attributes[4],1) # MONK-1 dataset where a5=1
monk1_2=select(m.monk1,m.attributes[4],2) # MONK-1 dataset where a5=2
monk1_3=select(m.monk1,m.attributes[4],3) # MONK-1 dataset where a5=3
monk1_4=select(m.monk1,m.attributes[4],4) # MONK-1 dataset where a5=4


# ATTRIBUTE A5 IN MONK-2
# A5 HAS VALUES {1,2,3,4}
    
monk2_1=select(m.monk1,m.attributes[4],1) # MONK-2 dataset where a5=1
monk2_2=select(m.monk1,m.attributes[4],2) # MONK-2 dataset where a5=2
monk2_3=select(m.monk1,m.attributes[4],3) # MONK-2 dataset where a5=3
monk2_4=select(m.monk1,m.attributes[4],4) # MONK-2 dataset where a5=4


# ATTRIBUTE A2 IN MONK-3
# A2 HAS VALUES {1,2,3}
    
monk3_1=select(m.monk1,m.attributes[1],1) # MONK-3 dataset where a2=1
monk3_2=select(m.monk1,m.attributes[1],2) # MONK-3 dataset where a2=2
monk3_3=select(m.monk1,m.attributes[1],3) # MONK-3 dataset where a2=3


# INFORMATION GAIN CALCULATION AFTER SPLITTING
# FOR MONK-1 SPLITTINGS

info_gain1_1=[0]*6
for i in range(6):
    info_gain1_1[i]=averageGain(monk1_1,m.attributes[i])
    print("Information gain in MONK-1 tree for a5=1 for a{} is {}".format(i+1,info_gain1_1[i]))
m1_1=mostCommon(monk1_1)
print("The most common output in MONK-1 for a5=1 is {}".format(m1_1))

info_gain1_2=[0]*6
for i in range(6):
    info_gain1_2[i]=averageGain(monk1_2,m.attributes[i])
    print("Information gain in MONK-1 tree for a5=2 for a{} is {}".format(i+1,info_gain1_2[i]))
m1_2=mostCommon(monk1_2)
print("The most common output in MONK-1 for a5=2 is {}".format(m1_2))

info_gain1_3=[0]*6
for i in range(6):
    info_gain1_3[i]=averageGain(monk1_3,m.attributes[i])
    print("Information gain in MONK-1 tree for a5=3 for a{} is {}".format(i+1,info_gain1_3[i]))
m1_3=mostCommon(monk1_3)
print("The most common output in MONK-1 for a5=3 is {}".format(m1_3))
    
info_gain1_4=[0]*6
for i in range(6):
    info_gain1_3[i]=averageGain(monk1_4,m.attributes[i])
    print("Information gain in MONK-1 tree for a5=4 for a{} is {}".format(i+1,info_gain1_4[i]))
m1_4=mostCommon(monk1_4)
print("The most common output in MONK-1 for a5=4 is {}".format(m1_4))
    
    
# FOR MONK-2 SPLITTINGS

info_gain2_1=[0]*6
for i in range(6):
    info_gain2_1[i]=averageGain(monk2_1,m.attributes[i])
    print("Information gain in MONK-2 tree for a5=1 for a{} is {}".format(i+1,info_gain2_1[i]))
m2_1=mostCommon(monk2_1)
print("The most common output in MONK-2 for a5=1 is {}".format(m2_1))

info_gain2_2=[0]*6
for i in range(6):
    info_gain2_2[i]=averageGain(monk2_2,m.attributes[i])
    print("Information gain in MONK-2 tree for a5=2 for a{} is {}".format(i+1,info_gain2_2[i]))
m2_2=mostCommon(monk2_2)
print("The most common output in MONK-2 for a5=2 is {}".format(m2_2))

info_gain2_3=[0]*6
for i in range(6):
    info_gain2_3[i]=averageGain(monk2_3,m.attributes[i])
    print("Information gain in MONK-2 tree for a5=3 for a{} is {}".format(i+1,info_gain2_3[i]))
m2_3=mostCommon(monk2_3)
print("The most common output in MONK-2 for a5=3 is {}".format(m2_3))
    
info_gain2_4=[0]*6
for i in range(6):
    info_gain2_4[i]=averageGain(monk2_4,m.attributes[i])
    print("Information gain in MONK-2 tree for a5=4 for a{} is {}".format(i+1,info_gain2_4[i]))
m2_4=mostCommon(monk2_4)
print("The most common output in MONK-2 for a5=4 is {}".format(m2_1))
    

# FOR MONK-3 SPLITTINGS

info_gain3_1=[0]*6
for i in range(6):
    info_gain3_1[i]=averageGain(monk3_1,m.attributes[i])
    print("Information gain in MONK-3 tree for a2=1 for a{} is {}".format(i+1,info_gain3_1[i]))
m3_1=mostCommon(monk3_1)
print("The most common output in MONK-3 for a2=1 is {}".format(m3_1))

info_gain3_2=[0]*6
for i in range(6):
    info_gain3_2[i]=averageGain(monk3_2,m.attributes[i])
    print("Information gain in MONK-3 tree for a2=2 for a{} is {}".format(i+1,info_gain3_2[i]))
m3_2=mostCommon(monk3_2)
print("The most common output in MONK-3 for a2=2 is {}".format(m3_2))

info_gain3_3=[0]*6
for i in range(6):
    info_gain3_3[i]=averageGain(monk3_3,m.attributes[i])
    print("Information gain in MONK-3 tree for a2=3 for a{} is {}".format(i+1,info_gain3_3[i]))
m3_3=mostCommon(monk3_3)
print("The most common output in MONK-3 for a2=3 is {}".format(m3_3))
    


# TREE BUILDING

import dtree as d
import drawtree_qt5 as d5

# FOR MONK-1
t1=d.buildTree(m.monk1, m.attributes);
print("The error in the training set is {} and the error in the testing set is {} for MONK-1".format(1-d.check(t1, m.monk1), 1-d.check(t1, m.monk1test)))

# FOR MONK-2
t2=d.buildTree(m.monk2, m.attributes);

print("The error in the training set is {} and the error in the testing set is {} for MONK-2".format(1-d.check(t2, m.monk2), 1-d.check(t2, m.monk2test)))

## FOR MONK-3
t3=d.buildTree(m.monk3, m.attributes);

print("The error in the training set is {} and the error in the testing set is {} for MONK-3".format(1-d.check(t3, m.monk3), 1-d.check(t3, m.monk3test)))


# TREE DRAWING
#d5.drawTree(t1)
#d5.drawTree(t2)
#d5.drawTree(t3)


# PRUNING

from partition import partition
import numpy as np

#def pruning(input_tree,validation):
#    error=1
#    aux=error
#    err=list()
#    while aux<=error:
#        aux=error
#        alt=d.allPruned(input_tree)
#        for i in range(len(alt)):
#            err.append(1-d.check(alt[i], validation))
#        print(err)
#        error=min(err)
#        ind=err.index(min(err))
#        input_tree=alt[ind]
#    return error




def pruning (p1,training,validation):
    validation_best = 1-d.check(p1[0], validation)
    for i in range(len(p1)):
        validation_current = 1-d.check(p1[i], validation)
        if (validation_current < validation_best) :
            validation_best = validation_current
    return validation_best




fraction=[0.3,0.4,0.5,0.6,0.7,0.8]
cont=0
mea=np.zeros(6)
stand=np.zeros(6)
for fr in fraction:
    error=[0]*100
    for i in range(100):
        monk3train, monk3val = partition(m.monk1, fr)
        p1 = allPruned(d.buildTree(monk3train, m.attributes))
        error[i]=pruning(p1,monk3train,monk3val)
    error=np.asarray(error)
    me=np.mean(error)
    st=np.std(error)
    mea[cont]=me
    stand[cont]=st
    cont+=1

print(mea)
print(stand)


import numpy as np
import matplotlib.pyplot as plt


fraction = [0.3,0.4,0.5,0.6,0.7,0.8]
means = mea
#plt.errorbar(fraction, means, yerr=stand)

plt.title('Classification Error on the test set of Monk 1')
plt.grid(True)
plt.plot(fraction,means,color='orange',marker='o')
plt.legend('Mean error')
plt.xlabel('Fractions')
plt.ylabel('Means of Error')   
plt.show()