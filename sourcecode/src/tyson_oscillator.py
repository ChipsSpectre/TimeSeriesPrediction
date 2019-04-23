import gillespy
import numpy as np

class Tyson2StateOscillator(gillespy.Model):
    """
    Here, as a test case, we run a simple two-state oscillator (Novak & Tyson 
    2008) as an example of a stochastic reaction system.

    Code from:
    https://github.com/JohnAbel/gillespy/blob/master/examples/tyson_oscillator.py
    """

    def __init__(self, parameter_values=None):
        system_volume = 300  # system volume
        gillespy.Model.__init__(
            self, name="tyson-2-state", volume=system_volume)
        self.timespan(np.linspace(0, 100, 101))
        # =============================================
        # Define model species, initial values, parameters, and volume
        # =============================================

        # Parameter values  for this biochemical system are given in
        # concentration units. However, stochastic systems must use population
        # values. For example, a concentration unit of 0.5mol/(L*s)
        # is multiplied by a volume unit, to get a population/s rate
        # constant. Thus, for our non-mass action reactions, we include the
        # parameter "vol" in order to convert population units to concentration
        # units.

        P = gillespy.Parameter(name='P', expression=2.0)
        kt = gillespy.Parameter(name='kt', expression=20.0)
        kd = gillespy.Parameter(name='kd', expression=1.0)
        a0 = gillespy.Parameter(name='a0', expression=0.005)
        a1 = gillespy.Parameter(name='a1', expression=0.05)
        a2 = gillespy.Parameter(name='a2', expression=0.1)
        kdx = gillespy.Parameter(name='kdx', expression=1.0)
        self.add_parameter([P, kt, kd, a0, a1, a2, kdx])

        # Species
        # Initial values of each species (concentration converted to pop.)
        X = gillespy.Species(
            name='X', initial_value=int(0.65609071*system_volume))
        Y = gillespy.Species(
            name='Y', initial_value=int(0.85088331*system_volume))
        self.add_species([X, Y])

        # =============================================
        # Define the reactions within the model
        # =============================================

        # creation of X:
        rxn1 = gillespy.Reaction(name='X production',
                                 reactants={},
                                 products={X: 1},
                                 propensity_function='vol*1/(1+(Y*Y/((vol*vol))))')

        # degradadation of X:
        rxn2 = gillespy.Reaction(name='X degradation',
                                 reactants={X: 1},
                                 products={},
                                 rate=kdx)

        # creation of Y:
        rxn3 = gillespy.Reaction(name='Y production',
                                 reactants={X: 1},
                                 products={X: 1, Y: 1},
                                 rate=kt)

        # degradation of Y:
        rxn4 = gillespy.Reaction(name='Y degradation',
                                 reactants={Y: 1},
                                 products={},
                                 rate=kd)

        # nonlinear Y term:
        rxn5 = gillespy.Reaction(name='Y nonlin',
                                 reactants={Y: 1},
                                 products={},
                                 propensity_function='Y/(a0 + a1*(Y/vol)+a2*Y*Y/(vol*vol))')

        self.add_reaction([rxn1, rxn2, rxn3, rxn4, rxn5])
