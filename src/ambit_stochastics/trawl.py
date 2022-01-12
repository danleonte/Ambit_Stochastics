"""A script for the simulation and inference of trawl processes"""

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
from helpers.input_checks  import check_temporal_positions
from helpers.input_checks  import check_levy_seed_params
from helpers.sampler  import gaussian_part_sampler 
from helpers.sampler  import jump_part_sampler

#from .helpers.loss_functions import qlike_func
#helper_module = import_file(os.path.join(Path().resolve().parent,'helpers','loss_functions'))

#qlike = helper_module.qlike_func
####################################

class trawl:
    def __init__(self,tau, nr_trawls, nr_simulations, trawl_function=None, decorrelation_time =-np.inf,\
                mesh_size = None, truncation_grid = None, truncation_cpp = None, 
                gaussian_part_params=None, jump_part_name=None, jump_part_params=None,
                cpp_part_name = None, cpp_part_params = None, cpp_intensity = None, values=None):
        '''a container class for trawl processes'''
         
        #shape of the ambit set and times by which we translate the ambit set
        check_temporal_positions(tau, nr_trawls, nr_simulations)
        self.tau                            = tau
        self.nr_trawls                      = nr_trawls
        self.nr_simulations                 = nr_simulations
        self.times                          = tau * np.arange(1,nr_trawls+1)
        self.decorrelation_time             = decorrelation_time
        
        self.trawl_function                 = trawl_function

        ### distributional parameters of the gaussian and jump parts of the levy seed 
        self.gaussian_part_params           = gaussian_part_params
        self.jump_part_name                 = jump_part_name
        self.jump_part_params               = jump_part_params

            
        ### arrays containing the gaussian, jump and cpp parts of the simulation
        self.gaussian_values          =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        self.jump_values              =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        self.cpp_values               =   np.empty(shape = [self.nr_simulations,self.nr_trawls])
        
        ### passed by the user or simulated using one of the simulation methods ### 
        ### grid, slice of cpp
        self.values                   =   values 
        
        #############################################################################     
        ### attributes required only for the slice partition simulation algorithm ###
        #check equations [to add] from [to add];
        
        self.I                    = None
        self.slice_areas_matrix             = None
        
        ################################################################## 
        ### attributes required only for the grid simulation algorithm ###
        
        self.truncation_grid          = truncation_grid       
        self.mesh_size                = mesh_size
        
        if isinstance(mesh_size,int) or isinstance(mesh_size,float):
            self.vol = mesh_size ** 2
        elif mesh_size == None:
             self.vol = None
        else:
            raise ValueError('please check the value of mesh_size')                     
        self.indicator_matrix               = None
        
        ##################################################################
        ### attributes required only for the cpp simulation algorithm  ###
        
        self.cpp_part_name                  = cpp_part_name
        self.cpp_part_params                = cpp_part_params
        self.cpp_intensity                  = cpp_intensity
        
    ######################################################################
    ###  Simulation algorithms:      I grid, II slice, III cpp        ###
    ######################################################################
    
    
       ############################ I Grid ############################
    
    def grid_creation_1d(self,min_t,max_t):
        """creates a grid on \([0,\phi(0)] \\times [\\text{min_t}, \\text{max_t}]\). each cell is represented by bottom left corner
        to each cell we associate a sample from the gaussian and jump parts of the trawl process
        
        Returns:
          gaussian_values
          jump_values      
        """
            
        coords = np.mgrid[0:self.phi(0):self.mesh_size,min_t:max_t:self.mesh_size]
        x, t   = coords[0].flatten(), coords[1].flatten() 
        
        areas            = self.vol * np.ones([self.nr_simulations,len(t)])
        gaussian_values  = gaussian_part_sampler(self.gaussian_part_params,areas)
        jump_values      = jump_part_sampler(self.jump_part_params,areas,self.jump_part_name)
        
            
        return x,t,gaussian_values,jump_values
    
    
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
        
        \[\begin{pmatrix}
              a[0]  &  a[0] -a[1]  & a[0] -a[1]  &    a[0] -a[1] &  \ldots  & a[0] -a[1] & a[0] -a[1] & a[0]\\
              a[1]  &  a[1] -a[2]  & a[1] -a[2]  &    a[1] -a[2]&  \ldots  & a[1] -a[2] & a[1]       & 0   \\
              a[2]  &  a[2] -a[3]  & a[2] -a[3]  &    a[2] -a[3]&  \ldots  & a[2]       & 0          & 0   \\
                    &              &             &              &  \vdots  &            &            &     \\ 
              a[k-2]& a[k-1]       & 0           &          0   &  \ldots  &  0         &       0    & 0   \\
              a[k-1]& 0            & 0           &          0   &          &  0         &       0    & 0   \\
        \end{pmatrix}
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

     
    def simulate(self,method,slice_convolution_type='diagonals'):
         assert method in {'cpp','grid','slice'},'simulation method not supported'
         
         if method == 'grid':
             pass
         
         elif method == 'slice':
             assert slice_convolution_type in {'fft','diagonals'}
         
             if self.decorrelation_time == -np.inf:
                self.compute_slice_areas_infinite_decorrelation_time()
                self.simulate_slice_infinite_decorrelation_time(slice_convolution_type)
             
             elif self.decorrelation_time > -np.inf:
                  self.compute_slice_areas_finite_decorrelation_time()
                  self.simulate_slice_finite_decorrelation_time(slice_convolution_type)
                  
         else:
             pass
             
