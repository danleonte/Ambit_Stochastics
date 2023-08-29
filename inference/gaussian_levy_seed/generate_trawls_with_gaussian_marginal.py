
import numpy as np
from ambit_stochastics.trawl import trawl

def generate_gaussian_seed_trawls(tau, nr_simulations, nr_trawls, envelope, envelope_params, gaussian_part_params, np_seed):
    assert envelope in ['exponential', 'gamma', 'ig']
    assert isinstance(envelope_params, tuple) and isinstance(gaussian_part_params, tuple)
    np.random.seed(seed=np_seed)

    if envelope == 'exponential':

        assert len(envelope_params) == 1
        lambda_ = envelope_params[0]
        trawl_function = lambda x: lambda_ * np.exp(x * lambda_) * (x <= 0)

    elif envelope == 'gamma':

        assert len(envelope_params) == 2
        H, delta = envelope_params
        trawl_function = lambda x: H / delta * (1 - x / delta) ** (-H - 1) * (x <= 0)

    elif envelope == 'ig':

        assert len(envelope_params) == 2
        gamma, delta = envelope_params
        # total_area = gamma/delta change of varialbe ()**-0.5 = z
        trawl_function = lambda x: (delta / gamma) * (1 - 2 * x / gamma ** 2) ** (-0.5) * np.exp(
            delta * gamma * (1 - (1 - 2 * x / gamma ** 2) ** 0.5)) * (x <= 0)

    decorrelation_time = -np.inf
    jump_part_params = (0, 0)
    jump_part_name = None

    trawl_slice = trawl(nr_trawls=nr_trawls, nr_simulations=nr_simulations, trawl_function=trawl_function, tau=tau,
                        decorrelation_time=decorrelation_time, gaussian_part_params=gaussian_part_params,
                        jump_part_name=jump_part_name, jump_part_params=jump_part_params)

    trawl_slice.simulate('slice', 'diagonals')
    return trawl_slice