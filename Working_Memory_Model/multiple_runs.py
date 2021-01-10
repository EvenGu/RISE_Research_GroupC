#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 23:36:01 2020

@author: vivekchari
"""
from model_Network import working_memory_model
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from helper import ch_dir
from Synapse_Loss import calculate, isNaN
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import time
class run_days():
    def __init__(self):
        self.P = eval(open('parameters.txt').read())
        self.neurons = self.P['neurons_wm']
        self.DBS = self.P['DBS']
        self.synapse_dataframe = calculate(synapses = (self.neurons ** 2), day_step_size = 5, tot_days = 10000, 
                                      gimme_days = False, gimme_frames = True, export = False)
        self.days = self.P['days_to_simulate']
        self.cumulative_primary_dataframe = pd.DataFrame()
        self.cumulative_firing_dataframe = pd.DataFrame()
        self.cumulative_accuracy_dataframe = pd.DataFrame()
        self.run()  
    def plot(self):
        print('Plotting Data...')
        sns.set(context = 'paper')
        a = sns.tsplot(time = 'time', value = 'wm', data = self.cumulative_primary_dataframe, unit = 'trial', ci = 95, condition = 'day')
        a.set(xlabel='time (s)',ylabel='Decoded $\hat{cue}$ value')
        a.set(ylim = (0,1.1))
        plt.show(a)
        '''
        print('Plotting Data...')
        figure2, (ax3, ax4) = plt.subplots(1, 2)
        if len(self.cumulative_firing_dataframe.query("tuning=='strong'")) > 0:
            sns.tsplot(condition = 'day', time="time",value="firing_rate",unit="neuron-trial",
            ax=ax3,ci=95,data = self.cumulative_firing_dataframe.query("tuning=='strong'").reset_index())
            
        if len(self.cumulative_firing_dataframe.query("tuning=='nonpreferred'")) > 0:
            sns.tsplot(time = "time",value = "firing_rate",unit = "neuron-trial", 
            ax = ax4,ci = 95,data = self.cumulative_firing_dataframe.query("tuning=='nonpreferred'").reset_index(), condition = 'day')
        
        ax3.set(xlabel='time (s)',ylabel='Normalized Firing Rate',title='Preferred Direction', ylim = (0,350))
        ax4.set(xlabel='time (s)',ylim=(0,350),ylabel='',title='Nonpreferred Direction')
        plt.show(figure2)
        '''
        self.cumulative_accuracy_dataframe['correct'] = (self.cumulative_accuracy_dataframe['correct'] - self.cumulative_accuracy_dataframe['correct'].min())/(self.cumulative_accuracy_dataframe['correct'].max() - self.cumulative_accuracy_dataframe['correct'].min())
        c = sns.tsplot(time='time', value='correct', unit='trial', condition='day',data = self.cumulative_accuracy_dataframe, ci=95)
        c.set(xlabel = 'time(s)', ylabel = 'DRT accuracy percentage')
        c.set(ylim = (.3,1))
        plt.show(c)  
    def run(self):
        for day in self.days:
            t_p = []
            for i in range (7):
                t_p.append(self.synapse_dataframe.at[self.synapse_dataframe.set_index('Day Number').index.get_loc(day),str('L' + str(i))])
            for i in t_p:
                if isNaN(i[1]):
                    i[1] =0
            t_p = t_p[::-1]
            print(t_p)
            self.n = working_memory_model(transmission_proportion = t_p,day = day, DBS = self.DBS)
            primary_dataframe, firing_dataframe, df_correct = self.n.go()
            print('Appending to dataframe...')
            self.cumulative_primary_dataframe = self.cumulative_primary_dataframe.append(primary_dataframe, ignore_index=True)
            self.cumulative_firing_dataframe = self.cumulative_firing_dataframe.append(firing_dataframe, ignore_index=True)
            self.cumulative_accuracy_dataframe = self.cumulative_accuracy_dataframe.append(df_correct, ignore_index=True)
            
    def export(self):
        ch_dir(('Day %s to %s'%(self.days[0], self.days[-1]))) #directory to which data is saved
        self.cumulative_primary_dataframe.to_csv("Primary_dataframe.csv")
        self.cumulative_firing_dataframe.to_csv('Firing_dataframe.csv')
        self.cumulative_accuracy_dataframe.to_csv('Accuracy_dataframe.csv')

if __name__ == '__main__':
    t0 = time.time()
    a = run_days()
    a.export()
    t1 = time.time()
    print(t1 - t0)
        

                             
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
                                    
