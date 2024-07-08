#Import packages
import numpy as np
from pyeam import LCA
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

class RLEAM():
    """The Reinforcement Learning and Evidence Accumulator Model class.
    """

    def __init__(self, metadata,theta,task):
        """Initialize a RLEAM.

        ----------
        Metadata dictionary
        ----------
        drift_rate_model: string
            Class of drift rate model
        eam_model: string
            Class of accumulator model (either lca or race)
        n_par: int
            Number of participants
        
        ----------
        Parameters dictionary (i.e. theta array)
        ----------
        leak : float
            The leak term for lca model
        inh : float
            The lateral inhibition across accumulators for lca model
        noise_std : float
            Std of the noise term of the lca model
        ndt : float
            Non-decision time
        a : float
            Threshold value
        lr : float
            Learning rate

        ----------
        Task dictionary 
        ----------
        n_trials : int
            Number of trials  
        option_1_rewards : numpy array
            The list of possible rewards of option 1
        option_2_rewards : numpy array
            The list of possible rewards of option 2
        option_3_rewards : numpy array
            The list of possible rewards of option 3
        option_4_rewards : numpy array
            The list of possible rewards of option 4
        trial_class : dict
            Dictionary with the classes of rounds. There are two main classes:
                - trials_2_options : numpy array
                    An array with the all the possible combinations with 2 options    
                - trials_3_options : numpy_array
                    An array with the all the possible combinations with 3 options   

        """

        # Get metadata
        self.eam_model = metadata['eam_model'] 
        self.drift_rate_model = metadata['drift_rate_model']
        self.n_par = metadata['n_par']
        
        # Get task information
        self.n_trials = task['n_trials']
        self.option_1_rewards = task['option_1_rewards']
        self.option_2_rewards = task['option_2_rewards']
        self.option_3_rewards = task['option_3_rewards']
        self.option_4_rewards = task['option_4_rewards']
        self.trial_class = task['trial_class']
        #self.trial_n_options = task['trial_n_options']
        self.experiment_trials = task['experiment_trials']
        
        #Generate prior reward
        min_reward = min(self.option_1_rewards)
        max_reward = max(self.option_4_rewards)

        prior_reward_distr = list(range(min_reward, max_reward + 1))

        self.prior_reward_std = np.std(prior_reward_distr)

        # Get paramters
        self.v_mod = theta['v_mod']
        self.v_mod_eng = theta['v_mod_eng']
        self.a = theta['a']
        self.lr = theta['lr']
        self.ndt = theta['ndt']
        self.leak = theta['leak']
        self.inh = theta['inh']
        self.noise_std = theta['noise_std']
        self.offset = theta['offset']

        # Call function to check that configuration is correct
        self._check_config()

    def _generate_trials(self):
        """Function to generate trials
        """
        class_2_options = self.trial_class['trials_2_options']
        class_3_options = self.trial_class['trials_3_options']

        class_2_list = class_2_options * self.factor
        class_3_list = class_3_options * self.factor

        joint_list = class_2_list + class_3_list

        shuffled_array = random.sample(joint_list, len(joint_list))
        
        trial_type = np.zeros((self.n_trials,2))
        
        for i in range(self.n_trials):
            arr = shuffled_array[i]
            if len(arr) == 2:
                trial_type[i,0] = 2
                for j in range(len(class_2_options)):
                    if arr == class_2_options[j]:
                        trial_type[i,1] = j

            if len(arr) == 3:
                trial_type[i,0] = 3
                for j in range(len(class_3_options)):
                    if arr == class_3_options[j]:
                        trial_type[i,1] = j

 
        return shuffled_array , trial_type

    def _check_config(self):
        """Function to check that configuration is correct
        """
        if (self.n_trials % 10 == 0):
            self.factor = int(self.n_trials/10)
        else:
            print("Error: The number of trials (n_trials) must be a multiple of 10")

        if (self.n_par != len(self.experiment_trials)):
            print("Error: The length of experiment_trials list must be equal to number of participants (n_par)")
        
    def simulate(self):
        """Run RLEAM simulations
        
        Returns
        -------
        Dictionary. For each participant and trial, returns:
            -rts
            -choices
            -prediction_errors
            -best minus second best vals
            -Whether the best options (correct) was chosen. If yes it returns 1 if no, ir returns 0.

        """
        
        #Initialize list of values that we will return
        rts_list = np.zeros((self.n_par,self.n_trials))
        choices_list = np.zeros((self.n_par,self.n_trials))
        best_minus_second_best_list = np.zeros((self.n_par,self.n_trials))
        pred_error = np.zeros((self.n_par,self.n_trials))
        correct_choice = np.zeros((self.n_par,self.n_trials))
        trial_order = []

        for p in tqdm(range(self.n_par)):

            #Get the trials of this participant
            par_trials = self.experiment_trials[p]

            #Set Q values
            Q_values = np.zeros((self.n_trials,4))

            #Initialize Q values
            Q_values[0] = [4,4,4,4]
            
            #Set Q values std
            Q_values_std = np.zeros((self.n_trials,4))

            #Initialize Q values std
            Q_values_std[0] = [self.prior_reward_std,
                               self.prior_reward_std ,
                               self.prior_reward_std ,
                               self.prior_reward_std ]

            #Start trials loop
            for t in range(self.n_trials):

                #Retrieve number of options and trial type 
                n_options = par_trials[t,0]
                trial_type = int(par_trials[t,1])
        
                #If there are two options in this trial:
                if n_options == 2:
        
                    # Set current trial
                    current_trial = self.trial_class['trials_2_options'][trial_type]
                
                    # Select their current Q_values
                    Q_vals = [Q_values[t,current_trial[0] -1],
                            Q_values[t,current_trial[1] -1]]
                
                    # Define which options are the best and the worst
                    highest_Q = max(Q_vals) 
                
                    if highest_Q == Q_values[t,current_trial[0] -1]:
                        best_option = current_trial[0]
                        worst_option = current_trial[1]
                    else:
                        best_option = current_trial[1]
                        worst_option = current_trial[0]
            
                    # Compute drift rates based on Q values and drift rate model
                    if self.drift_rate_model == "q_vanilla":
                        v0 = self.v_mod * Q_values[t,best_option-1]
                        v1 = self.v_mod * Q_values[t,worst_option-1]

                    if self.drift_rate_model == "q_vanilla_cv":
                        v0 = self.v_mod * Q_values[t,best_option-1]/Q_values_std[t,best_option-1]
                        v1 = self.v_mod * Q_values[t,worst_option-1]/Q_values_std[t,worst_option-1]
                
                    # Calculate best minus second best Q values
                    best_minus_second_best_list[p,t] = Q_values[t,best_option-1] - Q_values[t,worst_option-1]
                
                    # Simulate a decision (evidence accumulation process)
                    if self.eam_model == "lca":
                        metadata_eam = {'n_alternatives':2, 'n_sims': 1,'dt':0.1, 'max_t':8000, 'tqdm': False}
                        theta_eam = {'a':self.a,
                                    'noise_std':self.noise_std,
                                    'leak':self.leak,
                                    'inh':self.inh,
                                    'offset': self.offset, 
                                    'ndt':self.ndt,
                                    'v0':v0,
                                    'v1':v1,}
                        sim_out = LCA(metadata_eam,theta_eam).simulate()
                        rt = sim_out['rts'][0]
                        winner = sim_out['choices'][0]
                        
                    if self.eam_model == "race":
                        #theta_eam = {'v0': v0, 'v1': v1, 'a': a,'t': ndt}
                        #sim_out = simulator(model = 'race_no_z_2', theta = theta,n_samples = 1)
                
                        rt = sim_out['rts'][0][0]
                
                        winner = sim_out['choices'][0][0]
                
                    if winner == 0:
                        choice = best_option
                        #The correct (best) choice was made
                        correct_choice[p,t] = 1 

                    if winner == 1:
                        choice = worst_option 
                
                #If there are three options in this trial:
                if n_options == 3:
        
                    # Set current trial
                    current_trial = self.trial_class['trials_3_options'][trial_type]

                    # Select their current Q_values
                    Q_vals = [Q_values[t,current_trial[0] -1],
                            Q_values[t,current_trial[1] -1],
                            Q_values[t,current_trial[2] -1]]
                    
                    # Define which options are the best and the worst
                    highest_Q = max(Q_vals) 
                    lowest_Q = min(Q_vals)
                    
                    if highest_Q == Q_values[t,current_trial[0] -1]:
                        best_option = current_trial[0]
                        
                        if lowest_Q == Q_values[t,current_trial[1] -1]:
                            worst_option = current_trial[1]
                            second_best_option = current_trial[2]
                            
                        else: 
                            worst_option = current_trial[2]
                            second_best_option = current_trial[1]
                            
                        
                    elif highest_Q == Q_values[t,current_trial[1] -1]:
                        best_option = current_trial[1]
                        if lowest_Q == Q_values[t,current_trial[0] -1]:
                            worst_option = current_trial[0]
                            second_best_option = current_trial[2]
                            
                        else: 
                            worst_option = current_trial[2]
                            second_best_option = current_trial[0]
                        
                    else: 
                        best_option = current_trial[2]
                        
                        if lowest_Q == Q_values[t,current_trial[0] -1]:
                            worst_option = current_trial[0]
                            second_best_option = current_trial[1]
                            
                        else: 
                            worst_option = current_trial[1]
                            second_best_option = current_trial[0]
                    
                    best_minus_second_best_list[p,t] =  Q_values[t,best_option-1] - Q_values[t,second_best_option-1]
                    
                    # Compute drift rates based on Q values and drift rate model
                    if self.drift_rate_model == "q_vanilla":

                        v0 = self.v_mod * Q_values[t,best_option-1]
                        v1 = self.v_mod * Q_values[t,second_best_option-1]
                        v2 = self.v_mod * Q_values[t,worst_option-1]
                    
                    if self.drift_rate_model == "q_vanilla_cv":

                        v0 = self.v_mod * Q_values[t,best_option-1]/Q_values_std[t,best_option-1]
                        v1 = self.v_mod * Q_values[t,second_best_option-1]/Q_values_std[t,second_best_option-1]
                        v2 = self.v_mod * Q_values[t,worst_option-1]/Q_values_std[t,worst_option-1]
                    

                    # Simulate a decision (evidence accumulation process)
                    if self.eam_model == "lca":
                        metadata_eam = {'n_alternatives':2, 'n_sims': 1,'dt':0.1, 'max_t':8000,'tqdm': False}
                        theta_eam = {'a':self.a,
                                    'noise_std':self.noise_std,
                                    'leak':self.leak,
                                    'inh':self.inh,
                                    'offset': self.offset, 
                                    'ndt':self.ndt,
                                    'v0':v0,
                                    'v1':v1,
                                    'v2':v2}
                        sim_out = LCA(metadata_eam,theta_eam).simulate()
                        rt = sim_out['rts'][0]
                        winner = sim_out['choices'][0]
                        
                    if self.eam_model == "race":
                        #theta = {'v0': v0, 'v1': v1, 'v2': v2,'a': a,'t': ndt}
                        #sim_out = simulator(model = 'race_no_z_3', theta = theta,n_samples = 1)
                
                        rt = sim_out['rts'][0][0]
                        winner = sim_out['choices'][0][0]
                    
                    if winner == 0:
                        choice = best_option 
                        #The correct (best) choice was made
                        correct_choice[p,t] = 1 
                    if winner == 1:
                        choice = second_best_option 
                    if winner == 2:
                        choice = worst_option
        
                #GAMBLE (once option is chosen get random reward from its distribution)
                
                #If 1 was chosen
                if choice == 1:
                    feedback = random.choice(self.option_1_rewards)
                        
                #If 2 was chosen
                if choice == 2:
                    feedback = random.choice(self.option_2_rewards)
                    
                #If 3 was chosen
                if choice == 3:
                    feedback = random.choice(self.option_3_rewards)
                
                #If 4 was chosen
                if choice == 4:
                    feedback = random.choice(self.option_4_rewards)
                    
                #UPDATE RULE (i.e. update the Q values and or Q value standard deviation)
                    
                #If it is not the last trial, update chosen option with prediction error
                if t != self.n_trials-1: 
                    
                    Q_values[t+1,0] = Q_values[t,0]
                    Q_values[t+1,1] = Q_values[t,1] 
                    Q_values[t+1,2] = Q_values[t,2]
                    Q_values[t+1,3] = Q_values[t,3]
                    
                    pred_error[p,t] = feedback - Q_values[t,choice-1]
                    
                    Q_values[t+1,choice-1] += self.lr*(feedback - Q_values[t,choice-1]) 

                    #Update also standard deviation

                    #print(Q_values_std[t])

                    Q_values_std[t+1,0] = np.std(Q_values[:t+1,0]) 
                    Q_values_std[t+1,1] = np.std(Q_values[:t+1,1]) 
                    Q_values_std[t+1,2] = np.std(Q_values[:t+1,2]) 
                    Q_values_std[t+1,3] = np.std(Q_values[:t+1,3]) 
                        
                rts_list[p,t] = rt
                choices_list[p,t] = choice

        sim_results = {
            'rts':rts_list,
            'choices':choices_list,
            'pred_error': pred_error,
            'best_minus_second_best': best_minus_second_best_list,
            'q_vals': Q_values,
            'correct_choice': correct_choice,
            'trial_order': trial_order
        }

        return sim_results
    
    def plot_average_rt_per_trial(self,sim_out):
        rts = sim_out['rts']

        mean_rts = np.average(rts,axis = 0)
        std_rts = np.std(rts,axis = 0)

        # Plot 
        fig, ax = plt.subplots()
        ax.errorbar(np.arange(1,self.n_trials+1), mean_rts, yerr=std_rts, fmt='o', markerfacecolor='blue', markeredgecolor='blue',
            ecolor='lightblue',capsize=1, capthick=1, elinewidth=1)

        ax.set_xlabel('Trial number')
        ax.set_ylabel('Reaction time (s)')
        ax.set_title('Average reaction time per trial')
        plt.show()

    def boxplot_rt_per_trial_class(self,sim_out):

        # Define rt lists
        rts_trials_1_vs_4 = []
        rts_trials_1_vs_3 = []
        rts_trials_2_vs_4 = []
        rts_trials_1_vs_2_vs_4 = []
        rts_trials_2_vs_3 = []
        rts_trials_1_vs_2_vs_3 = []
        rts_trials_1_vs_2 = []
        rts_trials_3_vs_4 = []
        rts_trials_1_vs_3_vs_4 = []
        rts_trials_2_vs_3_vs_4 = []
        
        trial_order = self.experiment_trials#sim_out['trial_order']
        rts = sim_out['rts']

        for p in range(len(trial_order)):

            #par_trial_order = trial_order[p]
            par_experiment_trials = self.experiment_trials[p]

            for t in range(self.n_trials):
                #par_current_trial = par_trial_order[t]
                par_current_trial = par_experiment_trials[t]

                if par_current_trial[0] == 2:
                    if par_current_trial[1] == 0:
                        rts_trials_1_vs_4.append(rts[p,t])
                    if par_current_trial[1] == 1:
                        rts_trials_1_vs_3.append(rts[p,t])
                    if par_current_trial[1] == 2:
                        rts_trials_2_vs_4.append(rts[p,t])
                    if par_current_trial[1] == 3:
                        rts_trials_2_vs_3.append(rts[p,t])
                    if par_current_trial[1] == 4:
                        rts_trials_1_vs_2.append(rts[p,t])
                    if par_current_trial[1] == 5:
                        rts_trials_3_vs_4.append(rts[p,t])
                    
                if par_current_trial[0] == 3:
                    if par_current_trial[1] == 0:
                        rts_trials_1_vs_3_vs_4.append(rts[p,t])
                    if par_current_trial[1] == 1:
                        rts_trials_2_vs_3_vs_4.append(rts[p,t])
                    if par_current_trial[1] == 2:
                        rts_trials_1_vs_2_vs_3.append(rts[p,t])
                    if par_current_trial[1] == 3:
                        rts_trials_1_vs_2_vs_4.append(rts[p,t])


        labels = ['1 vs 4 (Very easy)',
                  '1 vs 3 (Easy)',
                  '2 vs 4 (Easy)',
                  '1 vs 2 vs 4 (Easy)',
                  '2 vs 3 (Medium)',
                  '1 vs 2 vs 3 (Medium)',
                  '1 vs 2 (Hard)',
                  '3 vs 4 (Hard)',
                  '1 vs 3 vs 4 (Hard)',
                  '2 vs 3 vs 4 (Hard)']
    
        
        rts_lists = [rts_trials_1_vs_4,
                     rts_trials_1_vs_3,
                     rts_trials_2_vs_4,
                     rts_trials_1_vs_2_vs_4,
                     rts_trials_2_vs_3,
                     rts_trials_1_vs_2_vs_3,
                     rts_trials_1_vs_2,
                     rts_trials_3_vs_4,
                     rts_trials_1_vs_3_vs_4,
                     rts_trials_2_vs_3_vs_4]
         
        plt.boxplot(rts_lists, labels=labels)
        plt.title('Boxplot of reaction times per trial class')
        plt.ylabel('Reaction Times')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() 
        plt.show()

    def simulate_trial(self,agent_state,eng_level,trial):
        """Simulate a single trial

        Args:
            agent_state (dict): Dictionary with information about the current state of the agent
            eng_level (int): Number indicating the engagement level (0,1 or 2)
            trial (int): Number indicating the current trial

        Returns:
            sim_out: Dictionary with information about 
        """

        # Get current Q values
        Q_values = agent_state['q_vals'][trial]

        # Get the trial type
        exp_trial =  self.experiment_trials[0][trial]
        n_options = exp_trial[0]
        trial_type = int(exp_trial[1])

        if n_options == 2:
            current_trial = self.trial_class['trials_2_options'][trial_type]

            # Select their current Q_values of the options of the trial
            Q_vals = [Q_values[current_trial[0] -1],
                      Q_values[current_trial[1] -1]]
                
            # Define which options are the best and the worst
            highest_Q = max(Q_vals) 
                
            if highest_Q == Q_values[current_trial[0] -1]:
                best_option = current_trial[0]
                worst_option = current_trial[1]
            else:
                best_option = current_trial[1]
                worst_option = current_trial[0]
            
            # Compute drift rates based on Q values and engagement level
            v0 = self.v_mod_eng[eng_level] * Q_values[best_option-1]
            v1 = self.v_mod_eng[eng_level] * Q_values[worst_option-1]
        
            # Calculate best minus second best Q values
            best_minus_second_best = Q_values[best_option-1] - Q_values[worst_option-1]
                
            # Simulate a decision (evidence accumulation process)
            if self.eam_model == "lca":
                metadata_eam = {'n_alternatives':2, 'n_sims': 1,'dt':0.1, 'max_t':8000, 'tqdm': False}
                theta_eam = {'a':self.a,
                            'noise_std':self.noise_std,
                            'leak':self.leak,
                            'inh':self.inh,
                            'offset': self.offset, 
                            'ndt':self.ndt,
                            'v0':v0,
                            'v1':v1,}
                sim_out = LCA(metadata_eam,theta_eam).simulate()
                rt = sim_out['rts'][0]
                winner = sim_out['choices'][0]

            if winner == 0:
                choice = best_option
                #The correct (best) choice was made
                correct_choice = 1 

            if winner == 1:
                choice = worst_option 
                correct_choice = 0
                

        if n_options == 3:

            # Set current trial
            current_trial = self.trial_class['trials_3_options'][trial_type]
          
            # Select their current Q_values
            Q_vals = [Q_values[current_trial[0] -1],
                 Q_values[current_trial[1] -1],
                 Q_values[current_trial[2] -1]]
                    
            # Define which options are the best and the worst
            highest_Q = max(Q_vals) 
            lowest_Q = min(Q_vals)
                    
            if highest_Q == Q_values[current_trial[0] -1]:
                
                best_option = current_trial[0]
                
                if lowest_Q == Q_values[current_trial[1] -1]:
                    worst_option = current_trial[1]
                    second_best_option = current_trial[2]
                    
                else: 
                    worst_option = current_trial[2]
                    second_best_option = current_trial[1]
                    
                
            elif highest_Q == Q_values[current_trial[1] -1]:
                
                best_option = current_trial[1]
                
                if lowest_Q == Q_values[current_trial[0] -1]:
                    worst_option = current_trial[0]
                    second_best_option = current_trial[2]
                    
                else: 
                    worst_option = current_trial[2]
                    second_best_option = current_trial[0]
                
            else: 
                best_option = current_trial[2]
                
                if lowest_Q == Q_values[current_trial[0] -1]:
                    worst_option = current_trial[0]
                    second_best_option = current_trial[1]
                    
                else: 
                    worst_option = current_trial[1]
                    second_best_option = current_trial[0]
            
            best_minus_second_best = Q_values[best_option-1] - Q_values[second_best_option-1]
                    
            # Compute drift rates based on Q values and engagement level
            v0 = self.v_mod_eng[eng_level] * Q_values[best_option-1]
            v1 = self.v_mod_eng[eng_level] * Q_values[second_best_option-1]
            v2 = self.v_mod_eng[eng_level] * Q_values[worst_option-1]
        
            # Simulate a decision (evidence accumulation process)
            if self.eam_model == "lca":
                metadata_eam = {'n_alternatives':2, 'n_sims': 1,'dt':0.1, 'max_t':8000,'tqdm': False}
                theta_eam = {'a':self.a,
                            'noise_std':self.noise_std,
                            'leak':self.leak,
                            'inh':self.inh,
                            'offset': self.offset, 
                            'ndt':self.ndt,
                            'v0':v0,
                            'v1':v1,
                            'v2':v2}
                sim_out = LCA(metadata_eam,theta_eam).simulate()
                rt = sim_out['rts'][0]
                winner = sim_out['choices'][0]
                        
            if self.eam_model == "race":
                #theta = {'v0': v0, 'v1': v1, 'v2': v2,'a': a,'t': ndt}
                #sim_out = simulator(model = 'race_no_z_3', theta = theta,n_samples = 1)
        
                rt = sim_out['rts'][0][0]
                winner = sim_out['choices'][0][0]
            
            if winner == 0:
                choice = best_option 
                #The correct (best) choice was made
                correct_choice = 1 
            if winner == 1:
                choice = second_best_option 
                correct_choice = 0
            if winner == 2:
                choice = worst_option
                correct_choice = 0


        #GAMBLE (once option is chosen get random reward from its distribution)
                
        #If 1 was chosen
        if choice == 1:
            feedback = random.choice(self.option_1_rewards)
                
        #If 2 was chosen
        if choice == 2:
            feedback = random.choice(self.option_2_rewards)
            
        #If 3 was chosen
        if choice == 3:
            feedback = random.choice(self.option_3_rewards)
        
        #If 4 was chosen
        if choice == 4:
            feedback = random.choice(self.option_4_rewards)
            
        #UPDATE RULE (i.e. update the Q values)

        updated_Q = np.zeros(4)

        pred_error = feedback - Q_values[choice-1]
            
        #If it is not the last trial, update chosen option with prediction error
        if trial != self.n_trials-1: 
            
            updated_Q[0] = Q_values[0]
            updated_Q[1] = Q_values[1]
            updated_Q[2] = Q_values[2]
            updated_Q[3] = Q_values[3]
            
            updated_Q[choice-1] += self.lr*(pred_error) 

        sim_out = {
            'rt': rt,
            'choice': choice,
            'correct_choice': correct_choice,
            'best_minus_second_best': best_minus_second_best,
            'pred_error': pred_error,
            'updated_q_vals': updated_Q
        }

        return sim_out