import gillespy
import numpy as np

class Tyson2StateOscillator(gillespy.Model):
    """
    Here, as a test case, we run a simple two-state oscillator (Novak & Tyson 
    2008) as an example of a stochastic reaction system.

    Code from:
    https://github.com/JohnAbel/gillespy/blob/master/examples/tyson_oscillator.py
    """

    def __init__(self, timespan, parameter_values):
        system_volume = 300  # system volume
        gillespy.Model.__init__(
            self, name="tyson-2-state", volume=system_volume)
        self.timespan(np.linspace(0, timespan - 1, timespan))
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

        kt = gillespy.Parameter(name='kt', expression=parameter_values["kt"]) #  expression=20.0)
        kd = gillespy.Parameter(name='kd', expression=parameter_values["kd"]) # expression=1.0)
        a0 = gillespy.Parameter(name='a0', expression=parameter_values["a0"]) # expression=0.005)
        a1 = gillespy.Parameter(name='a1', expression=parameter_values["a1"]) # expression=0.05)
        a2 = gillespy.Parameter(name='a2', expression=parameter_values["a2"]) # expression=0.1)
        kdx = gillespy.Parameter(name='kdx', expression=parameter_values["kdx"]) # expression=1.0)
        self.add_parameter([kt, kd, a0, a1, a2, kdx])

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

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    params = [
        {
            "kt": 20.0,
            "kd": 1.0,
            "a0": 0.005,
            "a1": 0.05,
            "a2": 0.1,
            "kdx": 1.0
        }
    ]
    tyson_model = Tyson2StateOscillator(timespan=100, parameter_values = params[0])

    # =============================================
    # Simulate the mode and return the trajectories
    # =============================================
    # To set up the model, first create an empty model object. Then, add
    # species and parameters as was set up above.
    tyson_trajectories = tyson_model.run(show_labels=False)

    # =============================================
    # plot just the first trajectory, 0, in both time and phase space:
    # =============================================
    from matplotlib import gridspec

    gs = gridspec.GridSpec(1, 2)

    ax0 = plt.subplot(gs[0, 0])
    ax0.plot(tyson_trajectories[0][:, 0], tyson_trajectories[0][:, 1],
             label='X')
    ax0.plot(tyson_trajectories[0][:, 0], tyson_trajectories[0][:, 2],
             label='Y')
    ax0.legend()
    ax0.set_xlabel('Time')
    ax0.set_ylabel('Species Count')
    ax0.set_title('Time Series Oscillation')

    ax1 = plt.subplot(gs[0, 1])
    ax1.plot(tyson_trajectories[0][:, 1], tyson_trajectories[0][:, 2], 'k')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Phase-Space Plot')

    plt.tight_layout()
    plt.show()
