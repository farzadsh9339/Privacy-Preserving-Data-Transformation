# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 14:07:23 2019

@author: farzad
"""
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
from RUCA import Accuracy_P_RUCA , Accuracy_U_RUCA
from PCA import Accuracy_U_PCA , Accuracy_P_PCA
from MDR import Accuracy_U_MDR , Accuracy_P_MDR
from DCA import Accuracy_P_DCA , Accuracy_U_DCA
from Random_Projection import Accuracy_P_RP , Accuracy_U_RP
from Full_dimension import Accuracy_P_FD , Accuracy_U_FD




plt.figure(figsize=(15, 15))
plt.plot(Accuracy_P_RP,Accuracy_U_RP,'bs--', color='red',label='Random Projection',)
plt.plot(Accuracy_P_RUCA,Accuracy_U_RUCA,'*--' ,color='blue',label='RUCA')
plt.plot(Accuracy_P_PCA,Accuracy_U_PCA,'+--', color='green',label='PCA')
plt.plot(Accuracy_P_DCA,Accuracy_U_DCA,'^--' ,color='magenta',label='DCA')
plt.plot(Accuracy_P_MDR,Accuracy_U_MDR,'bs-- ',color='black',label='MDR')
plt.plot(Accuracy_P_FD,Accuracy_U_FD,' *--',color='yellow',label='Full Dimension')
plt.xlabel('Privacy Accuracy')
plt.ylabel('Utility Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.show()