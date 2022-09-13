import numpy as np
import properscoring as ps


def compute_CRPS_loss(true_values,predicted_values_dict):
    #predicted_values_dict[h] has shape [nr_simulations,nr_trawls,nr_samples]
    
    assert isinstance(predicted_values_dict,dict)
    assert isinstance(true_values,np.ndarray)
    assert len(true_values.shape) == 2
    
    
    result_dict = dict()
    
    for h in predicted_values_dict.keys(): 
        array = []
        for simulation_nr in range(true_values.shape[0]):
        
            true_values_to_use      = true_values[simulation_nr,h:]
            predicted_values_to_use = predicted_values_dict[h][simulation_nr,:-h,:]
        
            result = ps.crps_ensemble(observations = true_values_to_use, forecasts = predicted_values_to_use,
                                  weights=None, issorted=False,  axis=-1)
        
            array.append(np.mean(result))
            
        result_dict[h] = np.array(array)
        
    return result_dict
        
    
    