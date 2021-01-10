#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  1 21:54:34 2020

@author: vivekchari
"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from helper import make_cues, ch_dir, reset_gain_bias, primary_dataframe, firing_dataframe
import nengo
from nengo.rc import rc
from nengo.dists import Choice, Exponential, Uniform
from pathos.helpers import freeze_support 
#from nengo_extras import CollapsingGexfConverter
import scipy.stats as ss
nengo.rc.set('progress', 'progress_bar', 'nengo.utils.progress.TerminalProgressBar')
import pathos
from matplotlib.cm import get_cmap
from sklearn.preprocessing import normalize
class working_memory_model():
    '''
    A model of working memory based of the sDRT task.
    
    Proportion: The percentage of neurons in the working memory network affected by the synaptic degradation(Default = 0)
    
    Transmission: The transmission of the synapse (Default = 1)
                            n
    Decoder Equation: y(𝑡)= ∑d(𝑓𝑖 ) 𝑎𝑖(𝑥(𝑡))
                            𝑖=0
    '''
    def __init__(self, transmission_proportion = [], day = 0, DBS = False):
        print('Importing model parameters from parameters.txt')
        self.P = eval(open('parameters.txt').read())
        self.seed = self.P['seed']
        self.rng = np.random.RandomState(seed = self.seed)
        self.net_seed =  self.rng.randint(100)
        self.ens_seed =  self.rng.randint(100)
        self.con_seed =  self.rng.randint(100)
        self.sim_seed =  self.rng.randint(100)
        self.n_trials = self.P['n_trials']
        self.n_processes = self.P['n_processes']
        self.drug_type = str(self.P['drug_type'])
        self.decision_type=str(self.P['decision_type'])
        self.drugs=self.P['drugs']
        self.trials, self.perceived, self.cues = make_cues(self.P)
        self.P['timesteps'] = np.arange(0,int((self.P['t_cue']+ self.P['t_delay'])/self.P['dt_sample']))
        self.P['cues'] = self.cues
        self.P['perceived'] = self.perceived
        self.t_p = transmission_proportion 
        self.spiking_array = []
        self.accuracy_data = []
        self.neuron_type = nengo.LIF()
        self.memory_threshhold = 0
        self.ground_zero = 750
        #np.random.seed(82900)
        self.day = day
        self.DBS = DBS
        self.pulse_duration = self.P['DBS pulsewidth']
        self.amplitude = self.P['DBS amplitude']
        self.frequency = self.P['DBS frequency']

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
            
            self.cue = self.cue_scale * self.cues[self.trial]
            return self.cue_scale * self.cues[self.trial]
        else:
            self.cue = 0
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
            if value > 0.0: output = 1.0
            elif value < 0.0: output = -1.0
        elif self.decision_type == 'basal_ganglia':
            if x[0] > x[1]: output = 1.0
            elif x[0] < x[1]: output = -1.0
        return output 
    def BG_rescale(self,x): #rescales -1 to 1 into 0.3 to 1, makes 2-dimensional
        pos_x = 0.5 * (x + 1)
        rescaled = 0.4 + 0.6 * pos_x, 0.4 + 0.6 * (1 - pos_x)
        return rescaled
    def f_dec(self,x):
        value = (x[0] + x[1])
        if value > self.memory_threshhold and self.cue > 0:
            return 1
        elif value < (-1 * self.memory_threshhold) and self.cue < 0:
           return 1
        return 0
    def DBS_function(self,t):
        pulsewidth = self.pulse_duration
        period = 1/(self.frequency)
        amplitude = self.amplitude
        return (amplitude * np.where(((t % period <(pulsewidth))),1, 0))
    def arrange(self, array, m):
        q = array.tolist()
        q.sort(key = lambda x: abs(x - m))
        return np.asarray(q)
    def prob_selection(self, ens):
        affected_prop = 1 - self.t_p[-1][0]
        print(affected_prop)
        x = np.arange(ens.n_neurons)
        n_synapses_affected = min(int(np.sqrt((ens.n_neurons ** 2) * affected_prop)), int(np.sqrt(ens.n_neurons ** 2)))
        xU, xL = x + (.75*n_synapses_affected), x - (.75*n_synapses_affected)
        #prob = ss.norm.cdf(xU, loc = self.ground_zero, scale = 3) - ss.norm.cdf(xL,loc = self.ground_zero, scale = 3)
        prob = ss.poisson.cdf(k = xU, mu =self.ground_zero) - ss.poisson.cdf(k = xL,mu = self.ground_zero)
        for i in prob:
            if round(i,3) == 0:
                i = .001
        np.nan_to_num(prob, copy=False)
        prob = prob / prob.sum() #normalize the probabilities so their sum is 1
        np.nan_to_num(prob, copy=False, nan=0)
        if n_synapses_affected > (.5*ens.n_neurons):
            idx = np.random.choice(x, size = n_synapses_affected * 2, p = prob, replace = True)
        else:
            idx = np.random.choice(x, size = n_synapses_affected * 2, p = prob, replace = False)

        idx2 = idx[int(len(idx)/2):]
        idx = idx[:int(len(idx)/2)]
        return idx, idx2
    
    def degrade_synaptic_connection(self, sim, ens):
        print ('Building connection weight matrix...')
        weights = sim.data[self.wm_recurrent].weights / sim.data[ens].gain[:, None]
        affected_prop = 1 - self.t_p[-1][0]
        print ('Choosing %s synapses...'%(int((self.neurons_wm ** 2) * affected_prop)))
        idx, idx2 = self.prob_selection(ens)
        print ('Sorting matrices..')
        idx = self.arrange(idx,self.ground_zero)
        #np.random.shuffle(idx2)
        sorted_idx =[]
        if self.trial == 0:
            sns.set(context = 'paper')
            '''hc = ["#BF1D00", "#9F3726",'#7F524C', '#3F8798',"#1FA2BE", "#00BDE5"]
            z = np.random.choice(idx, size = int(len(idx) * .5), replace = False)   
            g = sns.rugplot(z, height = 1, axis = 'x', color = hc[0], lw = .5, alpha = .5)
            g.set(xlabel = 'Neuron #')
            g.set(xlim=(0, self.neurons_wm))'''
            '''
            sp  = sns.scatterplot(idx, idx2, color ='r')
            sp.set(xlim = (0,1500), ylim = (0,1500), xlabel = 'Neuron', ylabel = 'Neuron')
            plt.show()'''
            
        for i in range(len(self.t_p) - 1):
            idx = idx[:int(np.ceil(((self.neurons_wm ** 2) * self.t_p[i][0])/(len(idx2))))]
            sorted_idx.append(idx)
        print('Degrading weights...')
        sums = 0
        for i in range(len(self.t_p) - 1):
            for j in range(len(sorted_idx[i])):
                for k in range(len(idx2)):
                    weights[sorted_idx[i][j]][idx2[k]] = weights[sorted_idx[i][j]][idx2[k]] * self.t_p[i][1]
                    sums += 1
            if self.trial == 0:
                print ('Transmission = %s, proportion = %s'%(self.t_p[i][1],self.t_p[i][0]))
                #print(weights[sorted_idx[i]].shape)
        a = sns.distplot(weights, color = 'r')
        a.set(ylim = (0,100000))
        plt.savefig('dist_%s.png'%(self.day),bbox_inches='tight', dpi = 300)
        '''sample = weights[::8, ::8]
        sample = np.abs(sample)
        sample[:, [-1]] = normalize(sample[:, -1, None], norm='max', axis=0)
        plt.rcParams["axes.grid"] = False
        fig, ax = plt.subplots()
        cmap = get_cmap('inferno')
        cmap.set_over('#f9fc9c')
        cmap.set_bad('#290a5b')
        cmap.set_under('#290a5b')
        for param in ['figure.facecolor', 'axes.facecolor', 'savefig.facecolor']:
            plt.rcParams[param] = '#212946'  # bluish dark grey
        im = ax.imshow(sample, norm=matplotlib.colors.LogNorm(vmin=0.000000001, vmax = .00001), cmap = cmap)
        fig.colorbar(im,ax=ax)
        ax.set_title('Day %s'%(self.day))
        fig.savefig('heatmap day %s.png'%(self.day),bbox_inches='tight', dpi = 300)'''
        return weights

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

        with nengo.Network(seed = self.net_seed) as model:
            wm = nengo.Ensemble(self.neurons_wm, 2, neuron_type = self.neuron_type,seed = self.ens_seed, label = 'Working Memory')
            self.wm_recurrent = nengo.Connection(wm, wm, synapse = self.tau_wm, function = self.wm_recurrent_function, seed = self.con_seed, 
                                                 solver = nengo.solvers.LstsqL2(weights=True))
        with nengo.Simulator(model,dt = self.dt, seed = self.sim_seed) as sim:
            pass
        weights = self.degrade_synaptic_connection(sim, wm)
        
        with nengo.Network(seed = self.net_seed) as model:
            cue = nengo.Node(output = self.cue_function, label = 'Cue')
            time = nengo.Node(output = self.time_function, label = 'Time')
            inputs = nengo.Ensemble(self.neurons_inputs, 2, seed = self.ens_seed, label = 'Input Neurons')
            noise_wm_node = nengo.Node(output = self.noise_bias_function, label = 'Noise injection (WM node)')
            noise_decision_node = nengo.Node(output = self.noise_decision_function, label = 'Noise injection (decision node)')
            wm = nengo.Ensemble(self.neurons_wm, 2, neuron_type = self.neuron_type,seed = self.ens_seed, label = 'Working Memory')
            cor = nengo.Ensemble(1, 1, neuron_type = nengo.Direct(),seed = self.ens_seed, label = 'Accuracy sensor')
            dbs = nengo.Node(output = self.DBS_function,label = 'Deep Brain Stimulation Node', size_out=1)
            if self.decision_type == 'default':
                decision = nengo.Ensemble(self.neurons_decide, 2, seed = self.ens_seed, label = 'Decision Maker')
                
            elif self.decision_type == 'basal_ganglia':
                utilities = nengo.networks.EnsembleArray(self.neurons_inputs, n_ensembles = 2, seed = self.ens_seed, label = 'Utility network')
                BasalGanglia = nengo.networks.BasalGanglia(2, self.neurons_decide)
                decision = nengo.networks.EnsembleArray(self.neurons_decide, n_ensembles = 2, intercepts = Uniform(0.2, 1), encoders = Uniform(1,1), seed = self.ens_seed, label = 'Decision ensemble (Basal Ganglia')
                temp = nengo.Ensemble(self.neurons_decide, 2,neuron_type = self.neuron_type, seed = self.ens_seed)
                bias = nengo.Node([1] * 2, label = 'bias node')
            output = nengo.Ensemble(self.neurons_decide, 1, neuron_type = self.neuron_type, seed = self.ens_seed, label = 'Output')
        
            
            nengo.Connection(cue, inputs[0], synapse = None, seed = self.con_seed)
            nengo.Connection(time, inputs[1], synapse = None, seed = self.con_seed)
            nengo.Connection(inputs, wm, synapse = self.tau_wm, function=self.inputs_function, seed = self.con_seed)         
            self.wm_recurrent = nengo.Connection(wm.neurons, wm.neurons, synapse = self.tau_wm, seed = self.con_seed, transform = weights)
            nengo.Connection(noise_wm_node, wm.neurons, synapse = self.tau_wm, transform = np.ones((self.neurons_wm,1)) * self.tau_wm, seed = self.con_seed)
            
            if self.DBS:
                nengo.Connection(dbs, wm.neurons, synapse = 0, seed = self.con_seed, transform = np.ones((self.neurons_wm,1)))

            if self.decision_type == 'default':
                wm_to_decision = nengo.Connection(wm[0], decision[0], synapse = self.tau, seed = self.con_seed)
                nengo.Connection(noise_decision_node, decision[1], synapse = None, seed = self.con_seed)
                nengo.Connection(decision, output,function = self.decision_function, seed = self.con_seed)
                nengo.Connection(decision, cor, synapse = 0.025,function = self.f_dec, seed = self.con_seed)

                
            elif self.decision_type == 'basal_ganglia':
                wm_to_decision = nengo.Connection(wm[0], utilities.input, synapse = self.tau, function = self.BG_rescale, seed = self.con_seed)
                nengo.Connection(BasalGanglia.output, decision.input, synapse = self.tau, seed = self.con_seed)
                nengo.Connection(noise_decision_node, BasalGanglia.input,synapse = None, seed = self.con_seed) #added external noise?
                nengo.Connection(bias, decision.input, synapse = self.tau, seed = self.con_seed)
                nengo.Connection(decision.input, decision.output, transform=(np.eye(2)-1), synapse = self.tau/2.0, seed = self.con_seed)
                nengo.Connection(decision.output,temp, seed = self.con_seed)
                nengo.Connection(temp,output,function = self.decision_function, seed = self.con_seed, synapse = None)
                nengo.Connection(temp, cor, synapse = 0.2,seed = self.con_seed, function = self.f_dec)
            
            probe_wm = nengo.Probe(wm[0],synapse = 0.024, sample_every = self.dt_sample)
            probe_spikes = nengo.Probe(wm.neurons, 'spikes', sample_every = self.dt_sample)
            probe_output = nengo.Probe(output,synapse = None, sample_every = self.dt_sample)
            p_cor = nengo.Probe(cor, synapse = None, sample_every = self.dt_sample)
            #data_dir = ch_dir()
            #CollapsingGexfConverter().convert(model).write('model.gexf') 

        print('Running trial %s...\n' %(self.trial+1))
        with nengo.Simulator(model,dt = self.dt, seed = self.sim_seed) as sim:
            if self.drug_type == 'biophysical': 
                sim = reset_gain_bias(self.P, model, sim, wm, self.wm_recurrent, wm_to_decision, self.drug)
            sim.run(self.t_cue + self.t_delay)
            xyz = sim.data[probe_spikes]
            abc = np.abs(sim.data[p_cor])
            print('Constructing Dataframes...')
            df_primary = primary_dataframe(self.P, sim, self.drug,self.trial, probe_wm, probe_output, self.day)
            df_firing = firing_dataframe(self.P,sim,self.drug,self.trial, sim.data[wm], probe_spikes, self.day)
            
        return [df_primary, df_firing, abc, xyz]
                
    def multiprocessing(self):
        print("\nDay = %s, trials = %s,..." %(self.day,self.n_trials))
        pool = pathos.multiprocessing.ProcessingPool()
        freeze_support()
        exp_params = []
        for drug in self.drugs:
            for trial in self.trials:
                exp_params.append([self.decision_type, self.drug_type, drug, trial, self.seed, self.P])
        #self.df_list = []
        self.df_list = pool.map(self.run, exp_params)
        #for i in range(self.n_trials):self.df_list.append(self.run(exp_params[i]))
        
        for i in range(len(self.df_list)):
            self.accuracy_data.append(self.df_list[i][2])
            
        self.accuracy_data = np.array(self.accuracy_data)
        
        for i in range(len(self.df_list)):
            self.spiking_array.append((self.df_list[i][3]))
        self.spiking_array = np.array(self.spiking_array)
        
        primary_dataframe = pd.concat([self.df_list[i][0] for i in range(len(self.df_list))], ignore_index=True)
        firing_dataframe = pd.concat([self.df_list[i][1] for i in range(len(self.df_list))], ignore_index=True)

        return primary_dataframe, firing_dataframe
    def export(self):
        print('Exporting Data...')
        ch_dir('Pickled dataframes') #directory to export data to
        primary_dataframe.to_pickle('primary_data.pkl')
        firing_dataframe.to_pickle('firing_data.pkl')
        param_df = pd.DataFrame([self.P])
        param_df.reset_index().to_json('params.json',orient='records')

    def array_to_pandas(self,data, datatype, drug):
        n_trials = data.shape[0]
        n_timesteps = data.shape[1]
        columns = ('trial', 'drug', 'time', datatype, 'day')
        df = pd.DataFrame(columns=columns)
        for trial in range(n_trials):
            print('Adding trial %s, to %s...' %(trial + 1, datatype))
            df_time = []
            for t in range(n_timesteps):
                df_temp = pd.DataFrame(
                    [[trial, drug + ' (model)', t*0.01, data[trial][t][0], self.day]], columns=columns)
                df_time.append(df_temp)
                del df_temp
            df_trial = pd.concat(df_time, ignore_index=True)
            df = pd.concat([df, df_trial], ignore_index=True)
            del df_time
        return df

        
    def plot_data(self, primary_dataframe, firing_dataframe, ):
        print('Plotting Data...')
        sns.set(context = 'paper')
        a = sns.tsplot(time = 'time', value = 'wm', data = primary_dataframe, unit = 'trial', ci = 95)
        a.set(xlabel='time (s)',ylabel='Decoded $\hat{cue}$ value',
              title= "Number of trials = %s" %(self.n_trials))
        a.set(ylim = (0,1))
        plt.show(a)
        print('Plotting Data...')
        figure2, (ax3, ax4) = plt.subplots(1, 2)
        if len(firing_dataframe.query("tuning=='strong'")) > 0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial",ax=ax3,ci=95,data=firing_dataframe.query("tuning=='strong'").reset_index(), legend = False)
        
        if len(firing_dataframe.query("tuning=='nonpreferred'"))>0:
            sns.tsplot(time="time",value="firing_rate",unit="neuron-trial", ax=ax4,ci=95,data=firing_dataframe.query("tuning=='nonpreferred'").reset_index(), legend = False)
        
        ax3.set(xlabel='time (s)',ylabel='Normalized Firing Rate',title='Preferred Direction', ylim = (0,350))
        ax4.set(xlabel='time (s)',ylim=(0,250),ylabel='',title='Nonpreferred Direction')
        plt.show(figure2)
        
        
        c = sns.tsplot(time='time', value='correct', unit='trial', condition='drug',data = self.df_correct, ci=95, legend = False)
        c.set(xlabel = 'time(s)', ylabel = 'DRT accuracy percentage',title= "Number of trials = %s" %(self.n_trials))
        c.set(ylim = (.4,1.1))
        plt.show(c)

      
    def go(self):
        primary_dataframe, firing_dataframe = self.multiprocessing()
        
        df_correct = self.array_to_pandas(self.accuracy_data, 'correct', 'control')
        self.df_correct = pd.concat([df_correct], ignore_index=True)
        if __name__ == '__main__':
            self.plot_data(primary_dataframe, firing_dataframe)
        
        self.primary_dataframe = primary_dataframe
        
        self.firing_dataframe = firing_dataframe
        
        return primary_dataframe, firing_dataframe, self.df_correct

if __name__ == '__main__':
    a = working_memory_model(transmission_proportion=[[0.05607450486603623, 0.0], [0.010180959787289263, 0.4338728679641582], [0.008689042024563994, 0.5551691077625127], [0.00864204162692773, 0.6757675862041994], [0.008285658627653694, 0.8012765861052439], [0.007596906695402993, 0.9255125029748694], [0.8995308863721255, 1.0]])
    a.go()
    
