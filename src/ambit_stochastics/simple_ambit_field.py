###################################################################
#imports 
from collections     import Counter
from itertools       import chain
from scipy.optimize  import fsolve
from scipy.integrate import quad
import numpy as np
import math
import time
#from numba import njit

from .helpers.input_checks import check_trawl_function
from .helpers.input_checks  import check_jump_part_and_params
from .helpers.input_checks  import check_gaussian_params
from .helpers.input_checks import check_spatio_temporal_positions

from .helpers.sampler import gaussian_part_sampler
from .helpers.sampler import jump_part_sampler
###################################################################

class simple_ambit_field:
    def __init__(self, x, tau, k_s, k_t, nr_simulations, ambit_function=None, decorrelation_time=-np.inf,
                 gaussian_part_params=None, jump_part_name=None, jump_part_params=None,
                 batch_size=None, total_nr_samples=None, values=None):
        """Container class for the simulation, parameter inference and forecasting of ambit fields of the
        form \(Y_t(x) = L(A+(x,t))\).
        
        Args:
          x: positive number: spacing between ambit sets on the space axis.
          tau: positive number: spacing between ambit sets on the time axis.
          k_s: positive integer: number of ambit sets on the space axix.
          k_t: positive integer: number of ambit sets on the time axis.
          nr_simulation: positive integer: number of simulations.
          ambit_function: a non-negative, continuous, strictly increasing function
          \(\phi \colon (-\infty,0] \\to [0,\infty)\) with \(\phi(1) > 0, \phi(t) =0\) for \(t>0\).
          decorrelation_time: \(-\infty\) if the ambit set A is unbounded and finite, negative otherwise.
          gaussian_part_params: tuple with the mean and standard deviation of the Gaussian part.                                                           
          jump_part_name: tuple with the parameters of the jump part distribution check `helpers.sampler`
          for the parametrisation.
          jump_part_params: string: name of the jump part distribution. check `helpers.sampler` for the parametrisation.
          batch_size: positive integer: number of points to be used at once in the `approximate_slices` method,
          in order to optimise for cache memory.
          total_nr_samples: positive integer: total number of points to be used in the `approximate_slices` method.
          values: a numpy array with shape \([\\text{nr_simulations},k_s,k_t]\) which is passed by the user or 
          simulated with the method `simple_ambit_field.simulate`.
        """

        #################################################################################
        check_spatio_temporal_positions(x, tau, k_s, k_t, nr_simulations)
        self.x = x
        self.tau = tau
        self.k_s = k_s
        self.k_t = k_t
        self.nr_simulations = nr_simulations

        ### simulation parameters ###
        self.ambit_function = ambit_function
        self.gaussian_part_params = gaussian_part_params
        self.jump_part_name = jump_part_name
        self.jump_part_params = jump_part_params
        self.decorrelation_time = decorrelation_time
        self.total_nr_samples = total_nr_samples
        self.batch_size = batch_size

        ### dimension of the indicator matrix for each minimal slice 
        ### if decorrelation_time >  -inf, I_t = ceiling(decorrelation_time/tau)
        ### if decorrelation_time =  -inf, I_t = k_t - T/tau,
        ### where T = tau * floor{\phi^{-1}(x)/tau + 1} ###
        self.I_t = None
        self.I_s = None   #I_s = math.ceil(self.ambit_function(0)/self.x)


        ### minimal slices on t > T/tau ###
        self.unique_slices = None
        self.unique_slices_areas = None
        
        ### correction slices given by the intersections of ambit sets \(A_ij\) with \(1 \le i \le k_s, 1 \le j \le k_t\)
        ### with \(t < T/tau\), which we list from left to right in an array
        self.correction_slices_areas = None
        
        ### container for gaussian and jump parts. their sum is the result ###
        self.gaussian_values = None
        self.jump_values = None
        
        ### passed by the user or simulated using the simulate method 
        ### must have shape [nr_simulations,k_s,k_t] ###
        if values == None:
            self.values = None

        else:

            assert isinstance(
                values, np.ndarray), 'the values argument is not a numpy array'
            assert values.shape == (
                nr_simulations, k_s, k_t), 'please check the shape of the values argument'
            self.values = values

        #########################################################################################
        ### infered simulation parameters: skip this if you are only interested in simulations###
        self.inferred_parameters = None

        # self.inferred_parameters is a list with elements dictionaries of the form
        # {'inferred_ambit_function_name':   , inferred_ambit_function_params:    ,
        # 'inferred_gaussian_params':        ,'inferred_jump_params':              }

        # inferred_ambit_function        is 'exponential','gamma', 'ig' or a lambda function
        # inferred_ambit_function_params is a tuple
        # inferred_gaussian_params       is a tuple containing the mean and scale
        # inferred_jump_params           is a dictionary containing the name of the distribution
        # and its params, such as {'gamma': (1,1)}
        ##########################################################################################

    def delete_values(self):
        """Deletes the `values` attribute"""
        if self.values != None:
            self.values = None
            print('self.values has been deleted')
        #else:
            #print('no values to delete')

    def determine_slices_from_points(self, points_x, points_t):
        """Helper for the 'approximate_slices' method. Given random points with coordinates 
        `points_x` and `points_t` coordinates from the uniform distribution on \([x,x+\phi(0)] \\times [0,\\tau]\)
        which do not belong to \(A_{01}\) or \(A_{10}\), we check in which slice \(S\) of \(\mathcal{S}_{kl}\) each point is.
        we do so by using a 3d array with shape  \([\\text{nr_sampled_points},I_s,I_t]\) where the \(i^{\\text{th}}\) element \([i,:,:]\) is a matrix with \(\\text{kl}^{\\text{th}}\) element
        is a boolean given by \(x_i - x \cdot k < \phi(t_i -t \cdot l) \cdot (T < t_i-t \cdot l <0)\)
        where \((x_i,t_i)\)  are the coordinates of the \(i^{\\text{th}}\) uniform random sample. 
        [check description of the indicator here]
        
        Args:\n 
          points_x: x coordinates of the uniformly sampled points
          points_t: t coordinates of the uniformly sampled points
        
        Returns:
          a dictionary with keys given by tuples which represent the indicator matrices of 
          a minimal slice and values given by the number of points contained in the minimal slice
        """

        # coordinates at which the ambit field is simulated
        ambit_t_coords = self.tau * np.arange(1, self.I_t+1)
        ambit_x_coords = self.x * np.arange(1, self.I_s+1)
        
        x_ik = np.subtract.outer(points_x, ambit_x_coords)
        x_ikl = np.repeat(x_ik[:, :, np.newaxis], repeats=self.I_t, axis=2)
        t_il = np.subtract.outer(points_t, ambit_t_coords)
        phi_t_ikl = np.repeat(self.ambit_function(
            t_il)[:, np.newaxis, :], repeats=self.I_s, axis=1)
        range_indicator = x_ik > 0
        range_indicator = np.repeat(
            range_indicator[:, :, np.newaxis], repeats=self.I_t, axis=2)

        indicator = (x_ikl < phi_t_ikl) * range_indicator
        
        #in the unlikely case no minimal slice is identified
        if len(indicator) == 0:
            raise ValueError('use more samples in each batch')

        # we enumerate the unique indicator matrices together with the frequency counts
        # we change the shape from [total_nr_samples, I_s, I_t]  to [total_nr_samples, I_s * I_t]
        reshaped_indicator = indicator.reshape(indicator.shape[:-2]+(-1,))
        g = (tuple(i) for i in reshaped_indicator)
        return Counter(chain(g))

    def approximate_slices(self):
        """Identifies the minimal slices in \(S_{11}\) together with their areas and assigns these value to attributes
        `unique_slices` and `unique_slices_areas`."""
        
        print('Slice estimation procedure has started')
        start_time = time.time()

        
        if self.batch_size == None:
            self.batch_size = self.total_nr_samples // 10
        self.total_nr_samples = self.batch_size * (self.total_nr_samples // self.batch_size)

        # rectangle to simulate uniform rvs in space-time is [x, x + ambit_function(0)] x [0,tau]
        low_x, high_x = self.x, self.x + self.ambit_function(0)
        low_t, high_t = max(self.tau + self.decorrelation_time, 0), self.tau
        dict_ = dict()

        #use batches of points to optimise for cache memory and prevent memory overflow 
        for batch in range(self.total_nr_samples // self.batch_size):

            points_x = np.random.uniform(
                low=low_x, high=high_x, size=self.batch_size)
            points_t = np.random.uniform(
                low=low_t, high=high_t, size=self.batch_size)

            # throw away points not contained in A_11 = A + (x,tau):
            # (points_x,points_t) in A_11 if: 0 < points_x - x < phi(points_t - tau)
            # i.e. throw away points contained in ambit sets bottom or left of A_11
            # left of A_11: no such points, since we only simulate points with t coordinate > 0
            # bottom of A_11: must be in A_01; condition: (points_x < phi(points_t - tau)) *  (points_x  > 0)

            indicator_in_A_11 = (
                points_x - self.x < self.ambit_function(points_t - self.tau)) * (points_x - self.x > 0)
            indicator_in_A_01 = (points_x < self.ambit_function(
                points_t - self.tau)) * (points_x > 0)
            indicator_bottom_of_A_11 = indicator_in_A_11 * (~indicator_in_A_01)

            points_x = points_x[indicator_bottom_of_A_11]
            points_t = points_t[indicator_bottom_of_A_11]

            dict_to_add = self.determine_slices_from_points(points_x, points_t)

            for k, v in dict_to_add.items():
                if k in dict_:
                    dict_[k] += v
                else:
                    dict_[k] = v
                    
        # to add more diagnostics
        print('Slice estimation procedure has finished')
        end_time = time.time()
        print(end_time - start_time)
        
        percentage_points_kept = 100 * sum(dict_.values()) / self.total_nr_samples
        print(f"{round(percentage_points_kept,2)}% of points are used in the slice estimation")

        nr_unique_indicators = len(dict_.keys())
        self.unique_slices = np.array(list(dict_.keys())).reshape(
            nr_unique_indicators, self.I_s, self.I_t)
        
        self.unique_slices_areas = np.array(list(dict_.values())) * (high_x-low_x) * \
            (high_t - low_t) / self.total_nr_samples
            
            
    def determine_correction_slices(self,T):
        """Method to be used in the infinite decorrelation time to determine the areas of the
        intersection of the ambit sets at time coordinates \(\\tau,\ldots, k_t \\tau\) with 
        the region of the plane given by \(t < T\). The result is stored in the attribute
        `correction_slices_areas`.
        """
        
        self.correction_slices_areas = [quad(self.ambit_function, a = T - (i+1) * self.tau, b= T - i * self.tau , limit=500)[0]  
                                        for i in range(1,self.k_t)] + [quad(self.ambit_function,a= -np.inf,b=T - self.k_t,limit=500)[0]]
 
        
#    @njit
    def simulate_finite_decorrelation_time(self):
        """implementation of algorithm  [nr to be added] in [paper link]"""
        Y_gaussian = np.zeros((self.nr_simulations,self.k_s + 2 *self.I_s -2,self.k_t + 2 * self.I_t-2))
        Y_jump     = np.zeros((self.nr_simulations,self.k_s + 2 *self.I_s -2,self.k_t + 2 * self.I_t-2))
        for k in range(self.k_s + self.I_s -1):
            for l in range(self.k_t + self.I_t - 1):
                
                gaussian_to_add = np.zeros((self.k_s,self.k_t))
                jump_to_add     = np.zeros((self.k_s,self.k_t))
                
                #simulate S.
                for slice_S,area_S in zip(self.unique_slices,self.unique_slices_areas):
                    tiled_area_S = np.tile(area_S,(self.nr_simulations,1))
                    
                    gaussian_sample_slice = gaussian_part_sampler(self.gaussian_part_params,tiled_area_S)
                    jump_sample_slice     = jump_part_sampler(self.jump_part_params,tiled_area_S,self.jump_part_name)
                    
                    gaussian_to_add += slice_S * gaussian_sample_slice[:,:,np.newaxis]
                    jump_to_add     += slice_S * jump_sample_slice[:,:,np.newaxis]
                    
                Y_gaussian[:,k:k+self.I_s,l:l+self.I_t] +=  gaussian_to_add
                Y_jump[:,k:k+self.I_s,l:l+self.I_t]     +=  jump_to_add
 
        self.gaussian_values =  Y_gaussian[:,self.I_s-1:self.I_s+self.k_s-1,self.I_t-1:self.I_t+self.k_t-1]
        self.jump_values     =  Y_jump[:,self.I_s-1:self.I_s+self.k_s-1,self.I_t-1:self.I_t+self.k_t-1]

#    @njit
    def simulate_infinite_decorrelation_time(self,T):
        """implementation of algorithm  [nr to be added] in [paper link]"""

        assert T/self.tau == int(T/self.tau)
        T_tau = -int(T/self.tau)
        
        Y_gaussian = np.zeros((self.nr_simulations,self.k_s + 2 *self.I_s -2,2 * self.k_t + 2 * T_tau -1))
        Y_jump     = np.zeros((self.nr_simulations,self.k_s + 2 *self.I_s -2,2 * self.k_t + 2 * T_tau -1))
        
        #add correction slices
        self.determine_correction_slices(T)
        correction_slices_matrix   = np.tile(self.correction_slices_areas,(self.nr_simulations,self.k_s,1))
        gaussian_correction_slices = gaussian_part_sampler(self.gaussian_part_params,correction_slices_matrix)
        jump_correction_slices     = jump_part_sampler(self.jump_part_params,correction_slices_matrix,self.jump_part_name)
        
        #gaussian_correction_slices = np.fliplr(np.cumsum(np.fliplr(gaussian_correction_slices),axis=1))
        #jump_correction_slices     = np.fliplr(np.cumsum(np.fliplr(jump_correction_slices),axis=1))
        gaussian_correction_slices = (np.cumsum(gaussian_correction_slices[:,:,::-1],axis=2))[:,:,::-1]
        jump_correction_slices     = (np.cumsum(jump_correction_slices[:,:,::-1],axis=2))[:,:,::-1]

        
        Y_gaussian[:,self.I_s-1:self.I_s - 1+ self.k_s, T_tau:T_tau + self.k_t] +=  gaussian_correction_slices  
        Y_jump[:,self.I_s-1:self.I_s - 1+ self.k_s, T_tau:T_tau + self.k_t]     +=  jump_correction_slices  

        #implementation of algorithm [] from []
        for k in range(self.k_s + self.I_s -1):
            for l in range(self.k_t +  T_tau):

                gaussian_to_add = np.zeros((self.nr_simulations,self.I_s,self.I_t))
                jump_to_add     = np.zeros((self.nr_simulations,self.I_s,self.I_t))
                
                for slice_S,area_S in zip(self.unique_slices,self.unique_slices_areas):
                    tiled_area_S = np.tile(area_S,(self.nr_simulations,1))
                    #simulate S
                    
                    gaussian_sample_slice = gaussian_part_sampler(self.gaussian_part_params,tiled_area_S)
                    jump_sample_slice     = jump_part_sampler(self.jump_part_params,tiled_area_S,self.jump_part_name)
                    
                    gaussian_to_add = slice_S * gaussian_sample_slice[:,:,np.newaxis] 
                    jump_to_add     = slice_S * jump_sample_slice[:,:,np.newaxis] 
                    
                Y_gaussian[:,k:k+self.I_s,l:l+self.I_t] +=  gaussian_to_add
                Y_jump[:,k:k+self.I_s,l:l+self.I_t]     +=  jump_to_add
                    
                    
        self.gaussian_values = Y_gaussian[:,self.I_s-1:self.I_s - 1+ self.k_s,T_tau+1: T_tau+self.k_t+1]
        self.jump_values     = Y_jump[:,self.I_s-1:self.I_s-1+self.k_s,T_tau+1:T_tau+1+self.k_t]
                
    def simulate(self):
        """Simulate the ambit field at time coordinates \(\\tau,\ldots,k_t\\tau\) and space
        coordinates \(x,\ldots,k_s x\). The marginal law of this stationary
        process is given by the independent sum of the Gaussian and jump parts. See [] for an example
        and `helpers.sampler` for the parametrisations used. The simulated values are stored in the
        attribute `values`."""

        # checks
        self.delete_values()
        check_trawl_function(self.ambit_function)
        check_gaussian_params(self.gaussian_part_params)
        check_jump_part_and_params(self.jump_part_name,self.jump_part_params)



        #set I_s
        self.I_s = math.ceil(self.ambit_function(0)/self.x)
        # set I_t
        if self.decorrelation_time > -np.inf:    
            
            # decorrelation_time checks
            assert self.decorrelation_time < 0, 'please check the value of the decorrelation_time'
            assert self.ambit_function(self.decorrelation_time) == 0,\
                            'ambit_function(decorrelation_time) should be 0'
            self.I_t = math.ceil(-self.decorrelation_time/self.tau)
            
            

        elif self.decorrelation_time == -np.inf:
            
            T_tilde = fsolve(lambda t: self.ambit_function(t)-self.x,x0=-1)[0]
            T = self.tau * math.floor(1 + T_tilde/self.tau)
            assert (T/self.tau).is_integer()
            
            self.I_t = int(-T / self.tau) + self.k_t
            
        self.approximate_slices()
        
        if  self.decorrelation_time > -np.inf:           
            self.simulate_finite_decorrelation_time()
            
        elif self.decorrelation_time == -np.inf:
             #self.determine_correction_slices(T)
             self.simulate_infinite_decorrelation_time(T)


        
        self.values = self.gaussian_values + self.jump_values
