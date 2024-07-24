#Import packages
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class LCA():
    """The leaky competing accumulator class.
    """

    def __init__(self, metadata,theta):
        """Initialize a leaky competing accumulator.

        ----------
        Metadata in dictionary
        ----------
        n_alternatives : int
            Number of accumulators in the LCA
        n_sims : int
            Number of simulations for a given parameter combination theta
        n_max_t: int
            Maximum time that the simulation is allowed to run
        dt: float
            The dt / tao term, representing the time step size 
        tqdm: boolean
            If true then tqdm is used. If false the it is not used
        
        ----------
        Parameters in dictionary (i.e. theta array)
        ----------
        leak : float
            The leak term
        inh : float
            The lateral inhibition across accumulators 
        noise_std : float
            Std of the noise term of the LCA process
        ndt : float
            Non-decision time
        a : float
            Threshold value

        """

        #Metadata
        #self.metadata = #params['metadata']
        self.n_alternatives = metadata['n_alternatives']#params['n_alternatives']#self.metadata['n_alternatives']
        self.n_sims = metadata['n_sims']#params['n_sims']#self.metadata['n_sims']
        self.dt_tau = metadata['dt']#params['dt']#self.metadata['dt']
        self.sqrt_dt_tau = np.sqrt(self.dt_tau)
        self.use_tqdm = metadata['tqdm']

        #n_max_steps is the maximum number of simulation steps 
        # (i.e. if no accumulator crosses threshold, it will keep going until this val)
        self.n_max_steps = int(metadata['max_t']/self.dt_tau)#int(params['n_max_t']/self.dt_tau)#int(self.metadata['n_max_t']/self.dt_tau)
     
        #Load parameters 

        #theta = params['theta']
        if self.n_alternatives == 2:
            self.i0 = theta['i0']#params['v0']
            self.i1 = theta['i1']#params['v1']#
        
        if self.n_alternatives == 3:
            self.i0 = theta['i0']#params['v0']#theta['v0']
            self.i1 = theta['i1'] #params['v1']#theta['v1']
            self.i2 = theta['i2'] #params['v2']#theta['v2']

        self.a = theta['a']
        self.noise_std = theta['noise_std'] 
        self.leak = theta['leak']
        self.inh = theta['inh']
        self.ndt = theta['ndt']
        self.offset = theta['offset']

        self._check_model_config()

    def _check_model_config(self):
        if self.n_alternatives < 2:
            print("Error: you need at lest 2 alternative options")

        if self.n_alternatives > 3:
            print("Error: the maximum number of alternative choices is 3")

    def simulate(self):
        """Run LCA simulations
        the update formula:
            1. evidence =   prev evidence
                      + new_input (drift rate)
                      - leaked previous evidence
                      - weighted sum of evidences of other accumulators
                      + noise
            2. evidence <- output_bounding(evidence)

        Returns
        -------
        Dictionary
            rts and choices

        """

        rts = np.zeros(self.n_sims)
        choices = np.zeros(self.n_sims)

        #If there are two alternatives
        if self.n_alternatives == 2:

            if self.use_tqdm:

                for n in tqdm(range(self.n_sims)):
                    #Precompute noise:
                    noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

                    #Define the instantaneous evidence arrays
                    x0 = np.zeros(self.n_max_steps)
                    x1 = np.zeros(self.n_max_steps)

                    for t in range(self.n_max_steps):

                        if t == self.n_max_steps:
                            print("Max number of simulation steps reached. Consider increasing n_max_t")
                            break

                        #If first accumulator reaches threshold
                        if x0[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 0
                            break

                        #If second accumulator reaches threshold
                        if x1[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 1
                            break

                        #Update instantaneous evidence of the two accumulators
                        x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*x1[t])*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                        x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*x0[t])*self.dt_tau + noise[t,1]*self.sqrt_dt_tau

                        #Make sure that evidence is not negative (lower bound is zero)
                        x0[t+1] = np.clip(x0[t+1], 0, None)
                        x1[t+1] = np.clip(x1[t+1], 0, None)
            else:
                for n in range(self.n_sims):
                    #Precompute noise:
                    noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

                    #Define the instantaneous evidence arrays
                    x0 = np.zeros(self.n_max_steps)
                    x1 = np.zeros(self.n_max_steps)

                    for t in range(self.n_max_steps):

                        if t == self.n_max_steps:
                            print("Max number of simulation steps reached. Consider increasing n_max_t")
                            break

                        #If first accumulator reaches threshold
                        if x0[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 0
                            break

                        #If second accumulator reaches threshold
                        if x1[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 1
                            break

                        #Update instantaneous evidence of the two accumulators
                        x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*x1[t])*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                        x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*x0[t])*self.dt_tau + noise[t,1]*self.sqrt_dt_tau

                        #Make sure that evidence is not negative (lower bound is zero)
                        x0[t+1] = np.clip(x0[t+1], 0, None)
                        x1[t+1] = np.clip(x1[t+1], 0, None)
                
        #If there are three alternatives
        if self.n_alternatives == 3:

            if self.use_tqdm:

                for n in tqdm(range(self.n_sims)):
                    #Precompute noise:
                    noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

                    #Define the instantaneous evidence arrays
                    x0 = np.zeros(self.n_max_steps)
                    x1 = np.zeros(self.n_max_steps)
                    x2 = np.zeros(self.n_max_steps)

                    for t in range(self.n_max_steps):

                        if t == self.n_max_steps:
                            print("Max number of simulation steps reached. Consider increasing n_max_t")
                            break

                        #If first accumulator reaches threshold
                        if x0[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 0
                            break

                        #If second accumulator reaches threshold
                        if x1[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 1
                            break

                        #If third accumulator reaches threshold
                        if x2[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 2
                            break

                        #Update instantaneous evidence of the three accumulators
                        x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*(x1[t] + x2[t]))*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                        x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*(x0[t] + x2[t]))*self.dt_tau + noise[t,1]*self.sqrt_dt_tau
                        x2[t+1] = x2[t] + self.offset + (self.i2 + self.leak*x2[t] + self.inh*(x0[t] + x1[t]))*self.dt_tau + noise[t,2]*self.sqrt_dt_tau

                        #Make sure that evidence is not negative (lower bound is zero)
                        x0[t+1] = np.clip(x0[t+1], 0, None)
                        x1[t+1] = np.clip(x1[t+1], 0, None)
                        x2[t+1] = np.clip(x2[t+1], 0, None)
            
            else:

                for n in range(self.n_sims):
                    #Precompute noise:
                    noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

                    #Define the instantaneous evidence arrays
                    x0 = np.zeros(self.n_max_steps)
                    x1 = np.zeros(self.n_max_steps)
                    x2 = np.zeros(self.n_max_steps)

                    for t in range(self.n_max_steps):

                        if t == self.n_max_steps:
                            print("Max number of simulation steps reached. Consider increasing n_max_t")
                            break

                        #If first accumulator reaches threshold
                        if x0[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 0
                            break

                        #If second accumulator reaches threshold
                        if x1[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 1
                            break

                        #If third accumulator reaches threshold
                        if x2[t] >= self.a:
                            rts[n] = t*self.dt_tau + self.ndt
                            choices[n] = 2
                            break

                        #Update instantaneous evidence of the three accumulators
                        x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*(x1[t] + x2[t]))*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                        x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*(x0[t] + x2[t]))*self.dt_tau + noise[t,1]*self.sqrt_dt_tau
                        x2[t+1] = x2[t] + self.offset + (self.i2 + self.leak*x2[t] + self.inh*(x0[t] + x1[t]))*self.dt_tau + noise[t,2]*self.sqrt_dt_tau

                        #Make sure that evidence is not negative (lower bound is zero)
                        x0[t+1] = np.clip(x0[t+1], 0, None)
                        x1[t+1] = np.clip(x1[t+1], 0, None)
                        x2[t+1] = np.clip(x2[t+1], 0, None)

            
        sims_out = {'rts':rts, 'choices': choices}

        return sims_out

    def plot_simulations(self):
        """
        Plot reaction times and choices of LCA simulations
        """
        
        #Perform simulations
        sim_out = self.simulate()
        
        choices = sim_out['choices']
        rts = sim_out['rts']
        
        alpha_val = 0.4
        
        color_0 = "red"
        color_1 = "blue"
        color_2 = "green"
        
        #If there are two alternatives:
        if self.n_alternatives == 2:
            
            rts_choices_0 = []
            rts_choices_1 = []
        
            for i in range(self.n_sims):
                if choices[i] == 0:
                    rts_choices_0.append(rts[i]) 
                if choices[i] == 1:
                    rts_choices_1.append(rts[i]) 
                    
            #Plot histogram
            sequence_array = [rts_choices_0,rts_choices_1]
            
            joint_list = rts_choices_0 + rts_choices_1
            
            max_rt = max(joint_list)
            
            bins = np.linspace(0,max_rt,num = int(max_rt/self.dt_tau))
            
            values, bins, _ = plt.hist(sequence_array,density = True,bins = bins, stacked= True)
    
            plt.close() 

            counts_0 = len(rts_choices_0)
            counts_1 = len(rts_choices_1)

            p0 = counts_0/self.n_sims
            p1 = counts_1/self.n_sims

            fig, ax = plt.subplots()
            
            sum_vals_0 = np.sum(values[0])
            sum_vals_1 = np.sum(values[1])
            
            if sum_vals_0 <= 0.0001:
                bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                ax.fill_between(bins[:-1], bin_heights_1, color=color_1, alpha = alpha_val, label='Option 2')
            
            if sum_vals_1 <= 0.0001:    
                bin_heights_0 = values[0]*p0/(np.diff(bins)*sum_vals_0)
                ax.fill_between(bins[:-1], bin_heights_0, color=color_0, alpha = alpha_val, label='Option 1')            
            else:
                bin_heights_0 = values[0]*p0/(np.diff(bins)*sum_vals_0)
                bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                
                ax.fill_between(bins[:-1], bin_heights_0, color='red', alpha = alpha_val, label='Option 1')
                ax.fill_between(bins[:-1], bin_heights_1, color='green', alpha = alpha_val, label='Option 2')


            ax.set_xlabel('Reaction time (s)')
            ax.set_ylabel('Defective pdf')

            ax.set_title('Reaction times for '+str(self.n_sims)+' simulations')

            ax.legend()
            plt.show()

            print("Mean reaction time: " + str(np.mean(rts)) +" s")
        
        #If there are three alternatives:
        if self.n_alternatives == 3:
            
            rts_choices_0 = []
            rts_choices_1 = []
            rts_choices_2 = []
        
            for i in range(self.n_sims):
                if choices[i] == 0:
                    rts_choices_0.append(rts[i]) 
                if choices[i] == 1:
                    rts_choices_1.append(rts[i]) 
                if choices[i] == 2:
                    rts_choices_2.append(rts[i]) 
                    
            #Plot histogram
            sequence_array = [rts_choices_0,rts_choices_1,rts_choices_2]
            
            joint_list = rts_choices_0 + rts_choices_1 + rts_choices_2
            
            max_rt = max(joint_list)
            
            bins = np.linspace(0,max_rt,num = int(max_rt/self.dt_tau))
            
            values, bins, _ = plt.hist(sequence_array,density = True,bins = bins, stacked= True)
    
            plt.close() 

            counts_0 = len(rts_choices_0)
            counts_1 = len(rts_choices_1)
            counts_2 = len(rts_choices_2)

            p0 = counts_0/self.n_sims
            p1 = counts_1/self.n_sims
            p2 = counts_2/self.n_sims

            fig, ax = plt.subplots()
            
            sum_vals_0 = np.sum(values[0])
            sum_vals_1 = np.sum(values[1])
            sum_vals_2 = np.sum(values[2])
            
            if sum_vals_0 <= 0.0001:
                if sum_vals_1 <= 0.0001:  
                    bin_heights_2 = values[2]*p2/(np.diff(bins)*sum_vals_2)
                    ax.fill_between(bins[:-1], bin_heights_2, color='red', alpha = alpha_val, label='Option 3')
                else:
                    bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                    bin_heights_2 = values[2]*p2/(np.diff(bins)*sum_vals_2)
                    ax.fill_between(bins[:-1], bin_heights_1, color='red', alpha = alpha_val, label='Option 2')
                    ax.fill_between(bins[:-1], bin_heights_2, color='red', alpha = alpha_val, label='Option 3')
                    
            
            if sum_vals_1 <= 0.0001:  
                if sum_vals_0 <= 0.0001: 
                    bin_heights_2 = values[2]*p2/(np.diff(bins)*sum_vals_2)
                    ax.fill_between(bins[:-1], bin_heights_2, color=color_2, alpha = alpha_val, label='Option 3')
                else:
                    bin_heights_0 = values[0]*p0/(np.diff(bins)*sum_vals_0)
                    bin_heights_2 = values[2]*p2/(np.diff(bins)*sum_vals_2)
                    ax.fill_between(bins[:-1], bin_heights_0, color=color_0, alpha = alpha_val, label='Option 1')
                    ax.fill_between(bins[:-1], bin_heights_2, color=color_2, alpha = alpha_val, label='Option 3')
                    
            if sum_vals_2 <= 0.0001:  
                if sum_vals_0 <= 0.0001: 
                    bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                    ax.fill_between(bins[:-1], bin_heights_1, color=color_1, alpha = alpha_val, label='Option 2')
                else:
                    bin_heights_0 = values[0]*p0/(np.diff(bins)*sum_vals_0)
                    bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                    ax.fill_between(bins[:-1], bin_heights_0, color=color_0, alpha = alpha_val, label='Option 1')
                    ax.fill_between(bins[:-1], bin_heights_1, color=color_1, alpha = alpha_val, label='Option 2')
                    
            else:
                bin_heights_0 = values[0]*p0/(np.diff(bins)*sum_vals_0)
                bin_heights_1 = values[1]*p1/(np.diff(bins)*sum_vals_1)
                bin_heights_2 = values[1]*p2/(np.diff(bins)*sum_vals_2)
                
                ax.fill_between(bins[:-1], bin_heights_0, color=color_0, alpha=alpha_val, label='Option 1')
                ax.fill_between(bins[:-1], bin_heights_1, color=color_1, alpha=alpha_val, label='Option 2')
                ax.fill_between(bins[:-1], bin_heights_2, color=color_2, alpha=alpha_val, label='Otpion 3')

            ax.set_xlabel('Reaction time (s)')
            ax.set_ylabel('Defective pdf')

            ax.set_title('Reaction times for '+str(self.n_sims)+' simulations')

            ax.legend()
            plt.show()
             
            print("Mean reaction time: " + str(np.mean(rts)) +" s")

    def plot_race(self):
        #If there are two alternatives
        if self.n_alternatives == 2:
             #Precompute noise:
            noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

            #Define the instantaneous evidence arrays
            x0 = np.zeros(self.n_max_steps)
            x1 = np.zeros(self.n_max_steps)

            for t in range(self.n_max_steps):

                if t == self.n_max_steps:
                    print("Max number of simulation steps reached. Consider increasing n_max_t")
                    break

                #If first accumulator reaches threshold
                if x0[t] >= self.a:
                    final_t = t
                    break

                #If second accumulator reaches threshold
                if x1[t] >= self.a:
                    final_t = t
                    break


                #Update instantaneous evidence of the three accumulators
                x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*x1[t])*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*x0[t])*self.dt_tau + noise[t,1]*self.sqrt_dt_tau

                #Make sure that evidence is not negative (lower bound is zero)
                x0[t+1] = np.clip(x0[t+1], 0, None)
                x1[t+1] = np.clip(x1[t+1], 0, None)

            plt.plot(x0[:final_t])
            plt.plot(x1[:final_t])
            plt.show()


        #If there are three alternatives
        if self.n_alternatives == 3:
            #Precompute noise:
            noise = np.random.normal(scale = self.noise_std, size = (self.n_max_steps,self.n_alternatives))

            #Define the instantaneous evidence arrays
            x0 = np.zeros(self.n_max_steps)
            x1 = np.zeros(self.n_max_steps)
            x2 = np.zeros(self.n_max_steps)

            for t in range(self.n_max_steps):

                if t == self.n_max_steps:
                    print("Max number of simulation steps reached. Consider increasing n_max_t")
                    break

                #If first accumulator reaches threshold
                if x0[t] >= self.a:
                    final_t = t
                    break

                #If second accumulator reaches threshold
                if x1[t] >= self.a:
                    final_t = t
                    break

                #If third accumulator reaches threshold
                if x2[t] >= self.a:
                    final_t = t
                    break

                #Update instantaneous evidence of the three accumulators
                x0[t+1] = x0[t] + self.offset + (self.i0 + self.leak*x0[t] + self.inh*(x1[t] + x2[t]))*self.dt_tau + noise[t,0]*self.sqrt_dt_tau
                x1[t+1] = x1[t] + self.offset + (self.i1 + self.leak*x1[t] + self.inh*(x0[t] + x2[t]))*self.dt_tau + noise[t,1]*self.sqrt_dt_tau
                x2[t+1] = x2[t] + self.offset + (self.i2 + self.leak*x2[t] + self.inh*(x0[t] + x1[t]))*self.dt_tau + noise[t,2]*self.sqrt_dt_tau

                #Make sure that evidence is not negative (lower bound is zero)
                x0[t+1] = np.clip(x0[t+1], 0, None)
                x1[t+1] = np.clip(x1[t+1], 0, None)
                x2[t+1] = np.clip(x2[t+1], 0, None)

            plt.plot(x0[:final_t], label= "Option 1")
            plt.plot(x1[:final_t], label= "Option 2")
            plt.plot(x2[:final_t], label= "Option 3")
            plt.axhline(y=1, color='black', linestyle='--',label = "Threshold")
            plt.ylabel("Accumulated evidence")
            plt.xlabel("Simulation steps")
            plt.title("Evidence accumulators race")
            plt.legend()
            plt.show()
