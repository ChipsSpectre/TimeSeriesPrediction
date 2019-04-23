from tyson_oscillator import Tyson2StateOscillator
import gillespy
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

import sys
print(sys.path)
sys.path[:0] = ['..']

if __name__ == '__main__':
    np.random.seed(0)
    tyson_model = Tyson2StateOscillator()

    # =============================================
    # Simulate the mode and return the trajectories
    # =============================================
    # To set up the model, first create an empty model object. Then, add
    # species and parameters as was set up above.
    tyson_trajectories = tyson_model.run(show_labels=False)
    print(tyson_trajectories[0], tyson_trajectories[0].shape)
    print("finished running.")

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
