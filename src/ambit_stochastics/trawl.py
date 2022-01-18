###################################
#module imports
from scipy.integrate import quad
import numpy as np
import pandas as pd
import math

#from scipy.optimize import minimize
from scipy.signal import convolve2d #flip the filter 
#from statsmodels.tsa.stattools import acf
#from scipy.stats import norm,gamma,cauchy,invgauss,norminvgauss,\
#                           geninvgauss,poisson 

#helper function imports
from .helpers.input_checks  import check1
from .helpers.input_checks  import check_grid_params
#from .helpers.input_checks  import check_cpp_params
from .helpers.input_checks  import check_trawl_function
from .helpers.input_checks  import check_jump_part_and_params
from .helpers.input_checks  import check_gaussian_params

 
from .helpers.sampler       import gaussian_part_sampler 
from .helpers.sampler       import jump_part_sampler
from .helpers.alternative_convolution_implementation import cumulative_and_diagonal_sums

#from .helpers.loss_functions import qlike_func
#helper_module = import_file(os.path.join(Path().resolve().parent,'helpers','loss_functions'))

#qlike = helper_module.qlike_func
####################################

class trawl:
    def __init__(self,nr_trawls, nr_simulations, trawl_function=None, tau = None, 
                decorrelation_time = -np.inf, mesh_size = None, times_grid = None, 
                truncation_grid = None, truncation_cpp = None, gaussian_part_params=None, 
                jump_part_name=None,jump_part_params=None,cpp_part_name = None, 
                cpp_part_params = None, cpp_intensity = None, values=None):
        """A container class for the simulation, parameter inference and forecasting of trawl processes.
        
        The implemented simulation algorithms are the grid, slice and cpp algorithms, as described in [paper link].
        
        The arguments required for the `simulate` method are `nr_trawls`,`nr_simulations`,
        `trawl_function` and `gaussian_part_params`. Further, the slice method requires `tau`,
        `decorrelation_time`,`jump_part_name`,`jump_part_params`, the grid method requires
        `mesh_size`,`times_grid`,`truncation_grid`,`jump_part_name`,`jump_part_params` and
        the cpp method requires `truncation_cpp,`,`cpp_part_name`, 
        `cpp_part_params`, `cpp_intensity`.
               
        Args:
          nr_simulation: positive integer: number of simulations.
          nr_trawls: positive integer: number of trawls on the time axis.
          trawl_function: a non-negative, continuous, strictly increasing function \(\phi \colon (-\infty,0] \\to [0,\infty)\)
          with \(\phi(1) =0, \phi(t) =0\) for \(t>0\).
          gaussian_part_params: tuple with the mean and standard deviation of the Gaussian Part.
          tau: positive number: spacing between ambit sets on the time axis, used in the slice simulation method.
          decorrelation_time: \(-\infty\) if the ambit set A is unbounded and finite, negative otherwise, used in the slice simulation method.
          jump_part_name: a
          jump_part_params: b
          truncation_grid: c
         """
       
        #general attributes
        check1(nr_trawls, nr_simulations)
        self.nr_trawls                      = nr_trawls
        self.nr_simulations                 = nr_simulations
        
        #attributes required for simulation
        self.trawl_function                 = trawl_function
        self.gaussian_part_params           = gaussian_part_params

        
        #############################################################################     
        ### attributes required only for the slice partition simulation algorithm ###  
        
        self.tau                            = tau
        self.decorrelation_time             = decorrelation_time
        self.I                              = None
        self.slice_areas_matrix             = None
        # distributional parameters of the gaussian and jump parts of the levy seed 
        # jump_part_name and jump_part_params are also requried for the grid method
        self.jump_part_name                 = jump_part_name
        self.jump_part_params               = jump_part_params
        
        ################################################################## 
        ### attributes required only for the grid simulation algorithm ###
        
        self.times_grid               = times_grid
        self.truncation_grid          = truncation_grid       
        self.mesh_size                = mesh_size
        self.times_grid               = times_grid
        self.vol                      = None
        
        #self.indicator_matrix               = None
        
        ################################################################## 
        ### attributes required only for the cpp simulation algorithm  ###
        
        self.cpp_part_name   = cpp_part_name 
        self.cpp_part_params = cpp_part_params
        self.cpp_intensity   = cpp_intensity
            
        ### arrays containing the gaussian, jump and cpp parts of the simulation
        self.gaussian_values          =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        self.jump_values              =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        self.cpp_values               =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        
        ### passed by the user or simulated using one of the simulation methods ### 
        self.values                   =   values 
        
        
        ##################################################################
        ### attributes required only for the cpp simulation algorithm  ###
        
        self.cpp_part_name                  = cpp_part_name
        self.cpp_part_params                = cpp_part_params
        self.cpp_intensity                  = cpp_intensity
        
    ######################################################################
    ###  Simulation algorithms:      I slice, II grid, III cpp         ###
    ######################################################################
    
    
       ########################### I Slice ###########################

    
    def compute_slice_areas_finite_decorrelation_time(self):
        """Computes the \(I \\times k\) matrix 
        
        
        corresponding to the areas of the slices 
        
                
        \[\\begin{bmatrix}
        L(S_{11})  & \\ldots & L(S_{1,k-1}) & L(S_{1k}) \\\\
        L(S_{21})  &  \\ldots &  L(S_{2,k-1}) & L(S_{2k}) \\\\
         \\vdots &       &  \\vdots  & \\vdots  \\\\
        L(S_{I1}) &  \\ldots &  L(S_{I,k-1}) & L(S_{I,k})
        \\end{bmatrix}
        \]"""
        self.I = math.ceil(-self.decorrelation_time/self.tau)
        
        s_i1 = [quad(self.trawl_function,a=-i *self.tau, b = (-i+1) * self.tau)[0]
                for i in range(1,self.I+1)]
        s_i2 = np.append(np.diff(s_i1[::-1])[::-1],s_i1[-1])
        
        right_column = np.tile(s_i2[:,np.newaxis],(1,self.nr_trawls-1))
        left_column  = left_column  = (np.array(s_i1))[:,np.newaxis]
        self.slice_areas_matrix  = np.concatenate([left_column,right_column],axis=1)
        
        #to add I-1 columns of length I zeros
        #check entire program here and comapre with previous versions
                
    
    def compute_slice_areas_infinite_decorrelation_time(self):
        """Computes the  \(k \\times k\) matrix  
        
        \[\\begin{bmatrix}
              a_0     &  a_0 - a_1  & a_0 - a_1  &    \\ldots  & a_0 - a_1  & a_0 - a_1  & a_0 \\\\
              a_1     &  a_1 - a_2  & a_1 - a_2  &    \\ldots  & a_1 - a_2  & a_1        & 0   \\\\
              a_2     &  a_2 - a_3  & a_2 - a_3  &    \\ldots  & a_2        & 0          & 0   \\\\
                      &             &            &    \\vdots  &            &            &     \\\\
              a_{k-2} & a_{k-1}     & 0          &    \\ldots  &  0         & 0          & 0   \\\\
              a_{k-1} & 0           & 0          &             &  0         & 0          & 0
        \\end{bmatrix}\]
            
        corresponding to the areas of the slices 
        
                
        \[\\begin{bmatrix}
        L(S_{11})  & \\ldots & L(S_{1k,-1}) & L(S_{1k}) \\\\
        L(S_{21})  &  \\ldots &  L(S_{2,k-1}) & 0 \\\\
         \\vdots &       &  \\vdots  & \\vdots  \\\\
        L(S_{k1}) &  \\ldots &  0 & 0
        \\end{bmatrix}
        \]
         
        
        where \(k =\) `self.nr_trawls` and 
        
        \[\\begin{align}
        a_0 &= \int_{-\\tau}^0 \phi(u)du, \\\\
            \\vdots & \\\\
        a_{k-2} &= \int_{(-k+1)\\tau} ^{(-k+2)  \\tau} \phi(u) du, \\\\
        a_{k-1} &= \int_{-\infty}^{(-k+1)\\tau} \phi(u)du.
        \\end{align} 
        \]
        """
        s_i1 =  [quad(self.trawl_function,a=-i *self.tau, b = (-i+1) * self.tau)[0]
                 for i in range(1,self.nr_trawls)] + [quad(self.trawl_function,a=-np.inf,
                                                           b=(-self.nr_trawls+1)*self.tau)[0]]
        
                                                           
        #  a[0] -a[1] ,a[1] -a[2], ... , a[k-2] - a[k-1] , 0
        differences   = np.append(np.diff(s_i1[::-1])[::-1],0) 

        left_column  = np.array(s_i1)[:,np.newaxis]  
        right_column = np.zeros((self.nr_trawls,1))
        #we reconstruct the elements on the secondary diagonal at the end
        
                                                     
        middle_matrix = np.tile(differences[:,np.newaxis],(1,self.nr_trawls-2))
        whole_matrix  = np.concatenate([left_column,middle_matrix,right_column],axis=1)
        whole_matrix_reversed   = np.triu(np.fliplr(whole_matrix), k=0)
        np.fill_diagonal(whole_matrix_reversed,s_i1)     

        self.slice_areas_matrix = np.fliplr(whole_matrix_reversed)
    
    def simulate_slice_finite_decorrelation_time(self,slice_convolution_type):
        """helper for the `simulate_slice` method"""
        
        filter_ = np.fliplr(np.tril(np.ones(self.I),k=0)) 
        zero_matrix  = np.zeros([self.I,self.I-1])

        for simulation_nr in range(self.nr_simulations):
            gaussian_slices = gaussian_part_sampler(self.gaussian_part_params,self.slice_areas_matrix) 
            jump_slices     = jump_part_sampler(self.jump_part_params,self.slice_areas_matrix,self.jump_part_name)
            
            if slice_convolution_type == 'fft':
        
                gaussian_slices = np.concatenate([zero_matrix,gaussian_slices],axis=1)
                jump_slices = np.concatenate([zero_matrix,jump_slices],axis=1)
        
                #matrix with 1's on and below the secondary diagonal
                #flip the filter to agree with np convention
                self.gaussian_values[simulation_nr,:] = convolve2d(gaussian_slices,filter_[::-1,::-1],'valid')[0]
                self.jump_values[simulation_nr,:]     = convolve2d(jump_slices,filter_[::-1,::-1],'valid')[0]
        
            elif slice_convolution_type == 'diagonals':
                
                self.gaussian_values[simulation_nr,:] = cumulative_and_diagonal_sums(gaussian_slices)
                self.jump_slices_cumulative_sum[simulation_nr,:] =  cumulative_and_diagonal_sums(jump_slices)
                
                
    def simulate_slice_infinite_decorrelation_time(self,slice_convolution_type):
        """Helper for the `simulate_slice` method."""

        zero_matrix     = np.zeros([self.nr_trawls,self.nr_trawls-1])
        filter_         = np.fliplr(np.tril(np.ones(self.nr_trawls),k=0)) 


        for simulation_nr in range(self.nr_simulations):
            
            gaussian_slices = gaussian_part_sampler(self.gaussian_part_params,self.slice_areas_matrix) 
            jump_slices     = jump_part_sampler(self.jump_part_params,self.slice_areas_matrix,self.jump_part_name)
            
            if slice_convolution_type == 'fft':
                gaussian_slices = np.concatenate([zero_matrix,gaussian_slices],axis=1)
                jump_slices     = np.concatenate([zero_matrix,jump_slices],axis=1)
                
                self.gaussian_values[simulation_nr,:] = convolve2d(gaussian_slices,filter_[::-1,::-1],'valid')[0]
                self.jump_values[simulation_nr,:]     = convolve2d(jump_slices,filter_[::-1,::-1],'valid')[0]
                
            elif slice_convolution_type == 'diagonals':
                
                self.gaussian_values[simulation_nr,:] = cumulative_and_diagonal_sums(gaussian_slices)
                self.jump_slices_cumulative_sum[simulation_nr,:] =  cumulative_and_diagonal_sums(jump_slices)
        
        raise ValueError('not yet implemented')    
        
    def simulate_slice(self,slice_convolution_type):
        """implements algorithm [] from []. `slice_convolution_type` can be either [to add]"""
        if self.decorrelation_time == -np.inf:
            self.compute_slice_areas_infinite_decorrelation_time()
            self.simulate_slice_infinite_decorrelation_time(slice_convolution_type)
             
        elif self.decorrelation_time > -np.inf:
            assert(self.trawl_function(self.decorrelation_time)) == 0,'please check decorrelation time' 
            self.compute_slice_areas_finite_decorrelation_time()
            self.simulate_slice_finite_decorrelation_time(slice_convolution_type)
        self.values = self.gaussian_values + self.jump_values

                      
       ############################ II Grid ############################
    
    def grid_creation(self,min_t,max_t):
        """Creates a grid on \([0,\phi(0)] \\times [\\text{min_t}, \\text{max_t}]\). Each cell is represented by 
        the coordinates of its bottom left corner. To each cell we associate a sample from each of the gaussian
        and jump parts of the trawl process.
        
        Returns:
          gaussian_values:  array with the Gaussian of the Levy basis evaluated over the cells  
          jump_values: array with the jump part of the Levy basis evaluated over the cells      
        """
            
        coords = np.mgrid[0:self.trawl_function(0):self.mesh_size,min_t:max_t:self.mesh_size]
        x, t   = coords[0].flatten(), coords[1].flatten() 
        
        areas            = self.vol * np.ones([self.nr_simulations,len(t)])
        gaussian_values  = gaussian_part_sampler(self.gaussian_part_params,areas)
        jump_values      = jump_part_sampler(self.jump_part_params,areas,self.jump_part_name)
        
            
        return x,t,gaussian_values,jump_values
    
    def grid_update(self,i,t,gaussian_values,jump_values):
        """Inputs the values of the Levy basis evaluated over the grid cells on \([\\tau_{i-1}+\\text{truncation_grid},\\tau_{i-1}] \\times [0,\\phi(0)]\),
        removes the values corresponding to cells with time coordinates less than \(\\tau_{i} + \\text{truncation_grid}\) and adds new samples
        for the levy basis evaluated over the grid cells with time coordinates in \([\\tau_{i-1},\\tau_i]\) (see figure). 
        Assumes that the consecutive ambit sets at times \(\\tau_{i-1},\\tau_i\) are not disjoint, i.e.
        \(\\tau_i + \\text{truncation_grid} < \\tau_{i-1}\).
        
        Args:
          i: index of the trawl to be simulated
          t: time coordinates of the cells of the grid on \([\\tau_{i-1},\\tau_{i-1}+\\text{truncation_grid}] \\times [0,\phi(0)]\)
          gaussian_values: gaussian values for the grid on \([\\tau_{i-1},\\tau_{i-1}+\\text{truncation_grid}] \\times [0,\phi(0)]\)
          jump_values: jump values for the grid on \([\\tau_{i-1},\\tau_{i-1}+\\text{truncation_grid}] \\times [0,\phi(0)\)
        
        Returns:
          gaussian_values: gaussian values for the grid cells on \([\\tau_{i},\\tau_{i}+\\text{truncation_grid}] \\times [0,\phi(0)]\)
          jump_values: jump values for the grid cells on \([\\tau_{i},\\tau_{i}+\\text{truncation_grid}] \\times [0,\phi(0)]\)                                                                                                                                                                                                
        """
          

        ind_to_keep       =  t >= (self.times_grid[i] + self.truncation_grid)
        t[~ind_to_keep]  += -self.truncation_grid
        
        areas             = self.vol * np.ones([self.nr_simulations,sum(~ind_to_keep)])
        #print(gaussian_values[:,~ind_to_keep].shape)
        #print(self.gaussian_part_sampler(areas).shape)
        gaussian_values[:,~ind_to_keep] = gaussian_part_sampler(self.gaussian_part_params,areas)
        jump_values[:,~ind_to_keep]     = jump_part_sampler(self.jump_part_params,areas,self.jump_part_name)  
        #print('ind to keep sum is ', ind_to_keep.sum())
        #print('gaussian is ',gaussian_values.shape)
        #print('non_gaussian_values ',non_gaussian_values[:,ind_to_keep].shape)
        #print('new_gaussian_values ',new_gaussian_values.shape)
        #print('t new shape is ',t.shape)
        #print('x shape is', x.shape)
        
        return t,gaussian_values,jump_values   
    
    
    def simulate_grid(self):
        """Simulate the trawl proces at times `self.times_grid`, which don't have to be
        equally distant, via the grid method."""
        for i in range(len(self.times_grid)):
             if (i==0) or (self.times_grid[i-1] <= self.times_grid[i] + self.truncation_grid):
                 #check that we are creating the grid for the first time or that 
                 #trawls at time i-1 and i have empty intersection
                 x,t,gaussian_values, jump_values = self.grid_creation(self.times_grid[i] + self.truncation_grid, self.times_grid[i])

                
             elif self.times_grid[i-1] > self.times_grid[i] + self.truncation_grid:
                 #check that we have non empty intersection and update the grid
                 t,gaussian_values,jump_values = self.grid_update(i,t,gaussian_values,jump_values)
            
             indicators = x < self.trawl_function(t-self.times_grid[i])
             #print(gaussian_values.shape,indicators.shape)
             self.gaussian_values[:,i]  = gaussian_values @ indicators
             self.jump_values[:,i]      = jump_values @ indicators
             
        self.values = self.gaussian_values + self.jump_values
       ########################### III cpp ###########################
       
    def simulate_cpp():
        """ text to be added"""
        pass

       ####################### simulate meta-method #######################
    def simulate(self,method,slice_convolution_type='diagonals'):
         """Function to simulate from the trawl function. Contains sanity checks
         for the simulation parameters and uses helper functions for each simulation
         method.
         
         Args:
           method: one of the strings cpp, grid or slice
           slice_convolution_type: if method is set to slice, this can be one of 
           the strings diagonals or ftt, depending on the way we add up the simulated
           slice. This argument is ignored if method is set to grid or cpp."""
        
         check_trawl_function(self.trawl_function)
         check_gaussian_params(self.gaussian_part_params)
         
         assert method in {'cpp','grid','slice'},'simulation method not supported'
         
         if method == 'grid':
             check_jump_part_and_params(self.jump_part_name,self.jump_part_params)
             check_grid_params(self.mesh_size,self.truncation_grid,self.times_grid)
             self.vol = self.mesh_size **2
             self.simulate_grid()
             
         elif method == 'cpp':
             self.simulate_cpp()
                 
         elif method == 'slice':
             assert slice_convolution_type in {'fft','diagonals'}
             check_jump_part_and_params(self.jump_part_name,self.jump_part_params)

             self.simulate_slice(slice_convolution_type)         
             
    def theoretical_acf(self,t_values):
        """Computes the theoretical acf of the trawl process
        
        Args:
          t_values: array of time values 
        
        Returns:
          d_acf: a dictionary of the type  \(t: \\text{corr}(X_0,X_t)\), where \(t\) ranges over the input array `t_values`. 
        """
        total_area = quad(self.trawl_function,a=-np.inf,b= 0)[0]
        d_acf=dict()
        for t in t_values:
            d_acf[t] = quad(self.trawl_function,a=-np.inf,b= -t)[0]/total_area
        return d_acf
                  

             
