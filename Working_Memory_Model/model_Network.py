#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:54:34 2020

@author: vivekchari
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from helper import make_cues, make_addon, ch_dir, empirical_dataframe, MySolver, reset_gain_bias, primary_dataframe, firing_dataframe, get_correct
from pathos.multiprocessing import ProcessingPool as Pool
import nengo
from nengo.rc import rc
from nengo.dists import Choice, Exponential, Uniform
from pathos.helpers import freeze_support 
from nengo.utils.matplotlib import rasterplot


class working_memory_model():
    def __init__(self):
        
        print 'Importing model parameters from parameters.txt'
        self.P = eval(open('parameters.txt').read())
        self.seed = self.P['seed']
        self.n_trials = self.P['n_trials']
        self.n_processes = self.P['n_processes']
        self.drug_type = str(self.P['drug_type'])
        self.decision_type=str(self.P['decision_type'])
        self.drugs=self.P['drugs']
        self.trials, self.perceived, self.cues = make_cues(self.P)
        self.P['timesteps'] = np.arange(0,int((self.P['t_cue']+ self.P['t_delay'])/self.P['dt_sample']))
        self.P['cues'] = self.cues
        self.P['perceived'] = self.perceived
        self.spiking_array = 0

    
    def noise_bias_function(self,t):
		if self.drug_type=='neural':
			return np.random.normal(self.drug_effect_neural[self.drug],self.noise_wm)
		else:
			return np.random.normal(0.0,self.noise_wm)
        
    def noise_decision_function(self, t):
        if self.decision_type == 'default':
            return np.random.normal(0.0,self.noise_decision)
        elif self.decision_type == 'basal_ganglia':
			return np.random.normal(0.0, self.noise_decision,size = 2)
        
    def inputs_function(self,x):
        return x * self.tau_wm
    
    def cue_function(self, t):
        if t < self.t_cue and self.perceived[self.trial] != 0:
            return self.cue_scale * self.cues[self.trial]
        else:
            return 0
        
    def time_function(self,t):
        if t > self.t_cue:
            return self.time_scale
        else:
            return 0
        

    def wm_recurrent_function(self,x):
		if self.drug_type == 'functional':
			return x * self.drug_effect_functional[self.drug]
		else:
			return x

    def decision_function(self,x): 
        output=0.0
        if self.decision_type=='default':
            value = x[0] + x[1]
            if value > 0.0:
                output = 1.0
            elif value < 0.0:
                output = -1.0
        elif self.decision_type == 'basal_ganglia':
            if x[0] > x[1]:
                output = 1.0
            elif x[0] < x[1]:
                output = -1.0
        return output 

    def BG_rescale(self,x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
		pos_x = 0.5 * (x + 1)
		rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
		return rescaled
   
    def run(self, params):
        self.decision_type = params[0]
        self.drug_type = params[1]
        self.drug = params[2]
        self.trial = params[3]
        self.seed = params[4]
        self.P = params[5]
        self.dt = self.P['dt']
        self.dt_sample = self.P['dt_sample']
        self.t_cue = self.P['t_cue']
        self.t_delay = self.P['t_delay']
        self.drug_effect_neural = self.P['drug_effect_neural']
        self.drug_effect_functional = self.P['drug_effect_functional']
        self.drug_effect_biophysical = self.P['drug_effect_biophysical']
        self.enc_min_cutoff = self.P['enc_min_cutoff']
        self.enc_max_cutoff = self.P['enc_max_cutoff']
        self.sigma_smoothing = self.P['sigma_smoothing']
        self.frac = self.P['frac']
        self.neurons_inputs = self.P['neurons_inputs']
        self.neurons_wm = self.P['neurons_wm']
        self.neurons_decide = self.P['neurons_decide']
        self.time_scale = self.P['time_scale']
        self.cue_scale = self.P['cue_scale']
        self.tau = self.P['tau']
        self.tau_wm = self.P['tau_wm']
        self.noise_wm = self.P['noise_wm']
        self.noise_decision = self.P['noise_decision']
        self.perceived = self.P['perceived']
        self.cues = self.P['cues']
        
        if self.drug_type == 'biophysical': 
            rc.set("decoder_cache", "enabled", "False") #don't try to remember old decoders
        else:
            rc.set("decoder_cache", "enabled", "True")
        
        print 'Building model...'
        with nengo.Network(seed = self.seed + self.trial) as model:
            print 'Creating Ensembles...'
            
            cue = nengo.Node(output = self.cue_function)
            time = nengo.Node(output = self.time_function)
            inputs = nengo.Ensemble(self.neurons_inputs, 2)
            noise_wm_node = nengo.Node(output = self.noise_bias_function)
            noise_decision_node = nengo.Node(output = self.noise_decision_function)
            wm = nengo.Ensemble(self.neurons_wm, 2)
            
            if self.decision_type == 'default':
                decision = nengo.Ensemble(self.neurons_decide, 2)
                
            elif self.decision_type == 'basal_ganglia':
                utilities = nengo.networks.EnsembleArray(self.neurons_inputs, n_ensembles = 2)
                BasalGanglia = nengo.networks.BasalGanglia(2, self.neurons_decide)
                decision = nengo.networks.EnsembleArray(self.neurons_decide, n_ensembles = 2, intercepts = Uniform(0.2, 1), encoders = Uniform(1,1))
                temp = nengo.Ensemble(self.neurons_decide, 2)
                bias = nengo.Node([1] * 2)
            output = nengo.Ensemble(self.neurons_decide, 1)
            
            print 'Buildiing Connections'
            
            nengo.Connection(cue, inputs[0], synapse = None)
            nengo.Connection(time, inputs[1], synapse = None)
            nengo.Connection(inputs, wm, synapse = self.tau_wm, function=self.inputs_function)
            wm_recurrent=nengo.Connection(wm, wm, synapse = self.tau_wm, function = self.wm_recurrent_function)
            nengo.Connection(noise_wm_node, wm.neurons, synapse = self.tau_wm, transform = np.ones((self.neurons_wm,1))*self.tau_wm)
            
            if self.decision_type == 'default':
                wm_to_decision = nengo.Connection(wm[0], decision[0], synapse = self.tau)
                nengo.Connection(noise_decision_node, decision[1], synapse=None)
                nengo.Connection(decision, output,function = self.decision_function)
                
            elif self.decision_type == 'basal_ganglia':
                wm_to_decision = nengo.Connection(wm[0], utilities.input, synapse = self.tau, function = self.BG_rescale)
                nengo.Connection(BasalGanglia.output, decision.input, synapse = self.tau)
                nengo.Connection(noise_decision_node, BasalGanglia.input,synapse = None) #added external noise?
                nengo.Connection(bias, decision.input, synapse = self.tau)
                nengo.Connection(decision.input, decision.output, transform=(np.eye(2)-1), synapse=self.tau/2.0)
                nengo.Connection(decision.output,temp)
                nengo.Connection(temp,output,function = self.decision_function)
                
            print 'Probing Network'
            probe_wm = nengo.Probe(wm[0],synapse = 0.01, sample_every = self.dt_sample)
            probe_spikes = nengo.Probe(wm.neurons, 'spikes', sample_every = self.dt_sample)
            probe_output = nengo.Probe(output,synapse=None, sample_every = self.dt_sample)
            
        print 'Running trial %s...' %(self.trial+1)
        with nengo.Simulator(model,dt = self.dt) as sim:
            if self.drug_type == 'biophysical': 
                sim = reset_gain_bias(self.P, model, sim, wm, wm_recurrent, wm_to_decision, self.drug)
                
            sim.run(self.t_cue + self.t_delay)
            abc = []
            xyz = sim.data[probe_spikes]
            abc.append(xyz)
            t = sim.trange(dt = self.dt_sample)
            abc.append(t)
            df_primary = primary_dataframe(self.P, sim, self.drug,self.trial, probe_wm, probe_output)
            df_firing = firing_dataframe(self.P,sim,self.drug,self.trial, sim.data[wm], probe_spikes)
        
        return [df_primary, df_firing, abc]
                
                
        
    def multiprocessing(self):
        
        print "decision_type=%s, trials=%s..." %(self.decision_type,self.n_trials)
        pool = Pool(nodes=self.n_processes)
        freeze_support()
        exp_params = []
        for drug in self.drugs:
            for trial in self.trials:
                exp_params.append([self.decision_type, self.drug_type, drug, trial, self.seed, self.P])
        
        self.df_list = pool.map(self.run, exp_params)
        #df_list= self.run(exp_params[0])
        self.spiking_array = self.df_list[0].pop()
        print 'Constructing Dataframes...'
        primary_dataframe = pd.concat([self.df_list[i][0] for i in range(len(self.df_list))], ignore_index=True)
        firing_dataframe = pd.concat([self.df_list[i][1] for i in range(len(self.df_list))], ignore_index=True)
        
        return primary_dataframe, firing_dataframe
    def export(self):
        print 'Exporting Data...'
        datadir = ch_dir()
        primary_dataframe.to_pickle('primary_data.pkl')
        firing_dataframe.to_pickle('firing_data.pkl')
        param_df = pd.DataFrame([self.P])
        param_df.reset_index().to_json('params.json',orient='records')
        
    def plot_data(self, primary_dataframe, firing_dataframe, ):
        print 'Plotting Data...'
        sns.set(context = 'paper')
        a = sns.tsplot(time = 'time', value = 'wm', data = primary_dataframe, unit = 'trial', ci = 95)
        a.set(xlabel='time (s)',ylabel='Decoded $\hat{cue}$ value',title="Decision type = %s, Number of trials = %s" %(self.decision_type,self.n_trials))
        plt.show(a)
        a.savefig('decoded_cue_vs_time.png')

        
        print 'Plotting Data...'
        spikes = self.spiking_array[0]
        spike_sum = np.sum(spikes,axis=0)
        indices = np.argsort(spike_sum)[::-1]
        top_spikes=spikes[:,indices[0:50]]
        b = rasterplot(self.spiking_array[1], top_spikes, use_eventplot = True)
        b = plt.gca()
        b.invert_yaxis()
        b.set(xlabel = 'time (s)', ylabel = 'neuron \nactivity $a_i(t)$')
        plt.show(b)
        b.savefig('raster_plot.png')

        
        print 'Plotting Data...'

    	figure2, (ax3, ax4) = plt.subplots(1, 2)
        if len(firing_dataframe.query("tuning=='strong'")) > 0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",ax=ax3,ci=95,data=firing_dataframe.query("tuning=='strong'").reset_index(), legend = False)
        
        if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial", ax=ax4,ci=95,data=firing_dataframe.query("tuning=='nonpreferred'").reset_index(), legend = False)
        
        ax3.set(xlabel='time (s)',xlim=(0.0,10.5),ylabel='Normalized Firing Rate',title='Preferred Direction')
        ax4.set(xlabel='time (s)',xlim=(0.0,10.5),ylim=(0,250),ylabel='',title='Nonpreferred Direction')
        
        figure2.savefig('firing_plots.png')
        
        
        
    def go(self):
        primary_dataframe, firing_dataframe = self.multiprocessing()
        self.plot_data(primary_dataframe, firing_dataframe)
        self.primary_dataframe = primary_dataframe
        self.firing_dataframe = firing_dataframe
        


n = working_memory_model()
n.go()



 