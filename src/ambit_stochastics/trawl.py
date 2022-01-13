"""A library for the analysis, simulation and inference of trawl processes"""

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
from .helpers.input_checks  import check_levy_seed_params
from .helpers.sampler  import gaussian_part_sampler 
from .helpers.sampler  import jump_part_sampler

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
        """A container class for trawl processes.
        
        description of the simulation methods.
        
        The arguments required for the `simulate` method are `nr_trawls`,`nr_simulations`,
        `trawl_function` and `gaussian_part_params`. Further, the slice method requires `tau`,
        `decorrelation_time`,`jump_part_name`,`jump_part_params`, the grid method requires
        `mesh_size`,`times_grid`,`truncation_grid`,`jump_part_name`,`jump_part_params` and
        the cpp method requires `truncation_cpp = None,`,`cpp_part_name`, 
        `cpp_part_params`, `cpp_intensity`.
        
        description of the forecast methods.
       
        Args:
          tau: positive number: spacing between ambit sets on the time axis.
          nr_simulation: positive integer: number of simulations.
          trawl_function: a non-negative, continuous, strictly increasing function \(\phi \colon (-\infty,0] \\to [0,\infty)\)
          with \(\phi(1) =0, \phi(t) =0\) for \(t>0\).
          decorrelation_time: \(-\infty\) if the ambit set A is unbounded and finite, negative otherwise.
          gaussian_part_params:                                                             
          jump_part_name:
          jump_part_params: 
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
        
        if isinstance(mesh_size,int) or isinstance(mesh_size,float):
            self.vol = mesh_size ** 2
        elif mesh_size == None:
             self.vol = None
                     
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
        """Computes the matrix of slice areas \(k =\) `self.nr_trawls`
        
        \[\\begin{bmatrix}
              a[0]  &  a[0] -a[1]  & a[0] -a[1]  &      \ldots  & a[0] -a[1] & a[0] -a[1] & a[0]\n
              a[1]  &  a[1] -a[2]  & a[1] -a[2]  &      \ldots  & a[1] -a[2] & a[1]       & 0   \n
              a[2]  &  a[2] -a[3]  & a[2] -a[3]  &      \ldots  & a[2]       & 0          & 0   \n
                    &              &             &      \\vdots  &            &            &     \n
              a[k-2]& a[k-1]       & 0           &      \ldots  &  0         &       0    & 0   \n
              a[k-1]& 0            & 0           &              &  0         &       0    & 0
        \\end{bmatrix}\]
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
        
        gaussian_slices = gaussian_part_sampler(self.gaussian_part_params,self.slice_areas_matrix) 
        jump_slices     = jump_part_sampler(self.jump_part_params,self.slice_areas_matrix,self.jump_part_name)
        
        for simulation_nr in range(self.nr_simulations):
            if slice_convolution_type == 'fft':
                zero_matrix  = np.zeros([self.I-1,self.I])
        
                gaussian_slices = np.concatenate([zero_matrix,gaussian_slices],axis=1)
                jump_slices = np.concatenate([zero_matrix,jump_slices],axis=1)
        
                #matrix with 1's on and below the secondary diagonal
                filter_ = np.fliplr(np.tril(np.ones(self.I),k=0)) 
                #flip the filter to agree with np convention
                self.gaussian_values[simulation_nr,:] = convolve2d(gaussian_slices,filter_[::-1,::-1],'valid')[0]
                self.jump_values[simulation_nr,:]     = convolve2d(jump_slices,filter_[::-1,::-1],'valid')[0]
        
            elif slice_convolution_type == 'diagonals':
                raise ValueError('not yet implemented')
        
        
    
    def simulate_slice_infinite_decorrelation_time(self,slice_convolution_type):
        gaussian_slices = gaussian_part_sampler(self.gaussian_part_params,self.slice_areas_matrix) 
        jump_slices     = jump_part_sampler(self.jump_part_params,self.slice_areas_matrix,self.jump_part_name)
        
        zero_matrix     = np.zeros([self.I-1,self.I])
        raise ValueError('not yet implemented')
        
        
        
       ############################ II Grid ############################
    
    def grid_creation_1d(self,min_t,max_t):
        """Creates a grid on \([0,\phi(0)] \\times [\\text{min_t}, \\text{max_t}]\). Each cell is represented by bottom left corner
        to each cell we associate a sample from the gaussian and jump parts of the trawl process
        
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
    
    def grid_update_1d(self,i,x,t,gaussian_values,jump_values):
        """Updates in the grid simulation procedure.
        
        Args:
          i:
          x:
          t:
          gaussian_values:
          jump_values:
              
        Returns:
          t:
          gaussian_values:
          jump_values:
          """

        ind_to_keep       =  t >= (self.times_grid[i] + self.truncation_grid)
        t[~ind_to_keep]  += -self.truncation_grid
        
        areas             = self.vol * np.ones([self.nr_simulations,sum(~ind_to_keep)])
        #print(gaussian_values[:,~ind_to_keep].shape)
        #print(self.gaussian_part_sampler(areas).shape)
        gaussian_values[:,~ind_to_keep] = self.gaussian_part_sampler(areas)
        jump_values[:,~ind_to_keep]     = self.jump_part_sampler(areas)  
        #print('ind to keep sum is ', ind_to_keep.sum())
        #print('gaussian is ',gaussian_values.shape)
        #print('non_gaussian_values ',non_gaussian_values[:,ind_to_keep].shape)
        #print('new_gaussian_values ',new_gaussian_values.shape)
        #print('t new shape is ',t.shape)
        #print('x shape is', x.shape)
        
        return t,gaussian_values,jump_values   
    
    
    def simulate_grid(self):
        """Simulate the trawl proces at times `self.times`, which don't have to be
        equally distant, via the grid method."""
        for i in range(len(self.times_grid)):
             if (i==0) or (self.times_grid[i-1] <= self.times_grid[i] + self.T):
                 #check that we are creating the grid for the first time or that 
                 #trawls at time i-1 and i have empty intersection
                 if self.d == 1:
                     x,t,gaussian_values, jump_values = self.grid_creation_1d(self.times_grid[i] + self.T, self.times_grid[i])
                 elif self.d > 1:
                     raise ValueError('not implemented')
                
             elif self.times_grid[i-1] > self.times_grid[i] + self.T:
                 #check that we have non empty intersection and update the grid
                 t,gaussian_values,jump_values = self.grid_update_1d(i,x,t,gaussian_values,jump_values)
            
             indicators = x < self.phi(t-self.times[i])
             #print(gaussian_values.shape,indicators.shape)
             self.gaussian_values[:,i]  = gaussian_values @ indicators
             self.jump_values[:,i]      = jump_values @ indicators
             
        self.values = self.gaussian_values + self.jump_values
       ########################### III cpp ###########################
       
    def simulate_cpp():
        """ text """
        pass

       ####################### simulate method #######################
    def simulate(self,method,slice_convolution_type='diagonals'):
         assert method in {'cpp','grid','slice'},'simulation method not supported'
         
         if method == 'grid':
             self.simulate_grid()
             
         elif method == 'cpp':
             self.simulate_cpp()
                 
         elif method == 'slice':
             assert slice_convolution_type in {'fft','diagonals'}
         
             if self.decorrelation_time == -np.inf:
                self.compute_slice_areas_infinite_decorrelation_time()
                self.simulate_slice_infinite_decorrelation_time(slice_convolution_type)
             
             elif self.decorrelation_time > -np.inf:
                  self.compute_slice_areas_finite_decorrelation_time()
                  self.simulate_slice_finite_decorrelation_time(slice_convolution_type)
                  

             
