#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 16:27:40 2020

@author: vivekchari
"""
import warnings
warnings.simplefilter(action='ignore', category = UserWarning) #the tsplot deprecated warnings annoy me :/ 
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context = 'paper')
from os import listdir
from os.path import isfile, join

data_dir = '/Users/vivekchari/Downloads/drugs_and_working_memory-master/WorkingMemory/data/Day 0 to 2400Mon Aug 10 15:33:20 2020'
paths = [f for f in listdir(data_dir) if isfile(join(data_dir, f))]
sns.color_palette("rocket_r", 4)
databases = []
for i in range(len(paths)):
    databases.append(pd.read_csv(os.path.join(data_dir, paths[i])))

a = databases[0]

flatui = ["#9b59b6", "#3498db", "#3F8798
", "#1FA2BE", "#00BDE5"]

try:
    c = sns.tsplot(time='time', value='correct', unit='trial', condition='day',data = a, ci=95, alpha = .7, color=sns.color_palette())
    c.set(xlabel = 'time(s)', ylabel = 'sDRT Accuracy Percentage')
    c.set(ylim = (.3,1.0))
    plt.show(c)
except UserWarning:
    pass