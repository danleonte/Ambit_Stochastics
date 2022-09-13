"""p
"""
import numpy as np
from scipy.integrate import quad
from scipy.stats  import poisson


def kernel(x_bar,t_bar):
    return 1+ 0.1 * t_bar

def generate_cpp_points(min_x,max_x,min_t,max_t,cpp_intensity):
    
    area_times_intensity = (max_x-min_x)*(max_t-min_t) * cpp_intensity  
    nr_points = poisson.rvs(mu = area_times_intensity)
    points_x = np.random.uniform(low = min_x, high = max_x, size = nr_points)
    points_t = np.random.uniform(low = min_t, high = max_t, size = nr_points)
    
    return points_x,points_t

def convex_envelope(f,low,high,nr_points):
        x   = np.linspace(low,high,nr_points)
        f_x = tuple(np.log(f(i)) for i in x)
        #print('f_x is ' + str(f_x) )
        #print(x)

        delta_x = x[1]- x[0]

        slopes     = np.array([(f_x[i+1] - f_x[i]) / delta_x for i in range(nr_points-1)])
        intercepts = np.array([f_x[i] - x[i] * slopes[i] for i in range(nr_points-1)])
        
        total_masses = (-np.exp(slopes * x[:-1] + intercepts) + np.exp(slopes * x[1:] + intercepts))/slopes
        #total_mass = np.sum(total_masses)
        
        #intercepts = intercepts - np.log(total_mass)
        
        return {(x[i],x[i+1]) : (slopes[i],intercepts[i]) for i in range(nr_points-1)}, total_masses #/ total_mass


def custom_sampler(a,b,epsilon,nr_points):
    """draws samples from the pdf of 
    a * e^{-b * x} / x on [epsilon,inf] """
    pdf = lambda x: a * np.exp(-b*x)  * (x >= epsilon )  / x
    l_1_mass = quad(pdf,epsilon,1)[0]
    b_1_mass = quad(pdf,1,np.inf)[0]
    if  np.random.uniform(low=0,high=1) <= l_1_mass / (l_1_mass + b_1_mass):
        
        #sample from branch less than 1. plan:
        #1)normalize pdf
        #2)create piecewise-constant exponential envelope and compute normalising constant
        #3)sample from normalised piece-wise constant exponential envelope
        #4)accept/reject step
        #5)put back the drift - not needed actually
        
        #1)
        pdf_l_1 = lambda x:  a * np.exp(-b*x) * (x >= epsilon ) * (x <= 1) / (x * l_1_mass)
        #2)
        assert epsilon < 1
        envelope_dict, interval_prob = convex_envelope(pdf_l_1,epsilon ,1,nr_points)
        normalised_interval_prob = interval_prob / np.sum(interval_prob)
        #3) sample from the envelope

        cumulative_prob = np.array([0] + list(np.cumsum(normalised_interval_prob)))
        #print( cumulative_prob)
        
        OK = False
        while OK == False:
            
            u = np.random.uniform(low=0,high=1)
            interval_index = np.argmax(cumulative_prob >u) - 1
            #print('interval_index is ' + str(interval_index))
            x_ = np.linspace(epsilon,1,nr_points)
            left,right = x_[interval_index],x_[interval_index+1]
            #print(envelope_dict)
            slope, intercept = envelope_dict[(left,right)]
            normalised_intercept = intercept - np.log(np.sum(interval_prob))
            #for u in [c,d], F^{-1}(u)    = 1/a * [log( exp(a*c+b) + a * (u - P(X<=c) )) - b]
            proposal = (1/slope)* (np.log(np.exp(slope * left + normalised_intercept) + slope *(u-cumulative_prob[interval_index]))-normalised_intercept)
            #print(u-cumulative_prob[interval_index])
            #4) accept/reject step
            if np.random.uniform(0,1) <=  pdf_l_1(proposal) / np.exp(slope*proposal+intercept):
                OK = True

        return proposal
        
    else:
        #sample from branch bigger than 1
        #pdf_b_1 = lambda x: a * np.exp(-b*x) / x * (x >= epsilon ) * (x>= 1) / b_1_mass
        
        OK= False
        while OK == False:
             #rejection sampling with exponential envelope, pdf given by b * exp(-b *x) on [1,infinity)
             proposal = 1 - np.log(1 - np.random.uniform(low=0,high=1)) / b #sample from a truncated exponential 
             u = np.random.uniform(low=0,high=1)
             # accept if u <= ratio of pdfs
             if u <= 1 / proposal:
                 OK = True
        return proposal
#def generate_associated_values(kernel_func,x_bar,t_bar)
        



if __name__ == '__main__':
    np.random.seed(34274239)
    tau = 0.5; nr_trawls = 250; T = -15; nr_simulations = 2;
    times = np.arange(tau, (nr_trawls+1) * tau, tau)
    #trawl_function
    lambda_ = 1;
    trawl_function = lambda t : lambda_ * np.exp(t * lambda_) * (t<=0)
    #levy seed params
    alpha,beta = (2,2)
    levy_seed_measure = lambda x : (x>=0.0000000001) * alpha * np.exp(-beta*x) / x
    #jump_truncation
    jump_T = 0.005
    result = np.zeros([nr_simulations,nr_trawls])
    nr_points = 10

    
    min_t = tau + T
    max_t = tau * nr_trawls
    min_x =  0
    max_x = trawl_function(0)
    
    cpp_intensity = quad(levy_seed_measure,jump_T,np.inf)[0]


    
    for simulation_nr in range(nr_simulations):
        
    
        points_x, points_t = generate_cpp_points(min_x = min_x, max_x = max_x, 
                    min_t = min_t, max_t = max_t, cpp_intensity = cpp_intensity)
        
        associated_values = np.zeros(len(points_x))
        
        for i in range(len(points_x)):
            associated_values[i]  =  custom_sampler(alpha,beta,jump_T,nr_points)     
            associated_values[i] *=  kernel(points_x[i],points_t[i])
                                
        #(x_i,t_i) in A_t if t < t_i and x_i < phi(t_i-t)
        indicator_matrix = np.tile(points_x[:,np.newaxis],(1,nr_trawls)) < \
                        trawl_function(np.subtract.outer(points_t, times))
        result[simulation_nr,:] = associated_values @ indicator_matrix
        #5) add back the drift: integral 0f a *  exp(-b*x)) between jump_T and 1
        #drift = alpha/beta * ( 1 - np.exp(-beta) )
        
    with open('gamma_part_for_gauss_gamma.npy', 'wb') as f:
        np.save(f, result)
            
    
    
                        