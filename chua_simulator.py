import time
import numpy as np
import pandas as pd
from os import path
from scipy.integrate import ode



class ChuaSimulator(object):


    def __init__(self, step_func, R):
        self._R = R
        self._step_func = step_func


    @classmethod
    def build(cls, R):
        G = 1 / R

        # capacitances
        C1 = 10.0**-8 # F
        C2 = 10.0**-7 # F
        # inductance
        L = 18.0 * 10**-3 # H

        # resistances inside the Chua diode
        R1 = 220.0 # Ohm
        R2 = 220.0 # Ohm
        R3 = 2200.0 # Ohm
        R4 = 22000.0 # Ohm
        R5 = 22000.0 # Ohm
        R6 = 3300.0 # Ohm

        # Satuation voltage
        E_sat = 9.0 # V

        m01 = 1 / R1
        m11 = -1 / R3
        BP1 = R3 / (R2 + R3) * E_sat

        m02 = 1 / R4
        m12 = -1 / R6
        BP2 = R6 / (R5 + R6) * E_sat

        m0 = m11 + m02
        m1 = m11 + m12

        qBP2 = m1*BP2
        qBP1 = m1*BP2+m0*(BP1-BP2)

        m_op = qBP1 / (E_sat - BP1)
        m_on = -m_op

        def v_i_char(v):
            ''' voltage-current characteristics of Chua diode '''
            abs_v = abs(v)
            if abs_v < BP2:
                # inner segment
                return m1*v 
            elif abs(v) < BP1:
                # outer segments
                if v > 0:
                    # positive side
                    return qBP2 + m0*(v-BP2)
                else:
                    # negative side
                    return -qBP2 + m0*(v+BP2)
            else:
                # decaying segments
                if v > 0:
                    # positive side
                    return qBP1 - m_op*(v-BP1)
                else:
                    # negative side
                    return -qBP1 + m_on*(v+BP1)

        def step_func(t, pt, dimmy):
            ''' step function for simulation, discreate version of Jacobian '''
            v_C1 = (G*(pt[1]-pt[0]) - v_i_char(pt[0])) / C1
            v_C2 = (G*(pt[0]-pt[1]) + pt[2]) / C2
            i_L = -pt[1] / L
            return [v_C1, v_C2, i_L]

        return ChuaSimulator(step_func, R)


    def run(self, IC, dt, t_max, ode_method):
        r = ode(self._step_func).set_integrator(ode_method)
        r.set_initial_value(IC, 0.0).set_f_params(1)

        num_steps = int(t_max / dt) + 2
        v_C1 = np.zeros(num_steps)
        v_C2 = np.zeros(num_steps)
        i_L = np.zeros(num_steps)

        v_C1[0], v_C2[0], i_L[0] = IC

        f = open(path.join('outputs', 'output_R%d_%d.csv') %
            ( int(self._R), int(time.time()) ) , 'w')

        idx = 1
        try:
            f.write('v_C1,v_C2,i_L,t\n')
            while r.successful() and r.t < t_max:
                r.integrate(r.t+dt)

                v_C1[idx], v_C2[idx], i_L[idx] = r.y[0:3]
                idx += 1

                f.write('%f,%f,%f' % tuple(r.y) + ',%f' % r.t + '\n')
        except Exception as e:
            # clean up file IO after error is signalled
            f.close() # file must be openned already
            raise e
        f.close()

        return pd.DataFrame({'v_C1': v_C1, 'v_C2': v_C2, 'i_L': i_L})