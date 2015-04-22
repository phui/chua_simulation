import numpy as np
from chua_simulator import ChuaSimulator


if __name__ == "__main__":
    # static configuration of integration
    t_max = 0.5 # second
    dt = 0.000001 # second
    IC = [-0.5, -0.2, 0.0]
    ode_method = 'dopri5'

    low = 1200.0 # modify this field to start simulation
    high = 2200.0 # modify this field to start simulation
    N = 100 # modify this field to start simulation

    # inclusive linspace call
    for R in np.linspace(low, high+(high-low)/N, N+1):
        ChuaSimulator.build(R).run(IC, dt, t_max, ode_method)