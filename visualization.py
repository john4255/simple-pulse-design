import matplotlib.pyplot as plt
import numpy as np

def plot_inits(dt):
    _, ax = plt.subplots(5,5)
    plt.subplots_adjust(
        left=0.1,    # left side of the subplots of the figure
        right=0.9,   # right side of the subplots of the figure
        bottom=0.1,  # bottom of the subplots of the figure
        top=0.9,     # top of the subplots of the figure
        wspace=0.4,  # width reserved for blank space between subplots
        hspace=0.6   # height reserved for white space between subplots
    )

    profs = [
        'FFprime',
        'pprime',
        'area',
        'elongation',
        'F',
        'g0',
        'j_bootstrap',
        'j_total',
        'magnetic_shear',
        'n_e',
        'n_i',
        'n_impurity',
        'p_ecrh_e',
        'p_ecrh_e',
        'p_generic_heat_e',
        'p_generic_heat_i',
        'Phi',
        'psi',
        'q',
        'sigma_parallel',
        'T_e',
        'T_i',
        'v_loop',
        'volume',
        'Z_eff'
    ]

    for i, prof in enumerate(profs):
        data = dt.profiles[prof][0].to_numpy()
        min_data = np.min(data)
        ax[i // 5][i % 5].set_title("{} (min={})".format(prof, '%.2E' % Decimal(min_data)))
        ax[i // 5][i % 5].plot(data)
        ax[i // 5][i % 5].set_xlabel("")

    plt.show()

def graph_sim(sim_vars, i=0):
    _, ax = plt.subplots(3, 3)

    j = 0
    for key, val in sim_vars.items():
        if type(val) is list or type(val) is np.ndarray:
            print(key)
            print(val)
            continue
        elif key in 'T_e T_i n_e n_i':
            continue
        elif len(val) > 0:
            ax[j // 3][j % 3].set_title("{}".format(key))
            ax[j // 3][j % 3].plot(val[i])
            j += 1
    plt.show()

def graph_var(sim_vars, var, step):
    fig, ax = plt.subplots(1, len(sim_vars[var]), figsize=(8, 4))
    fig.suptitle("{} at step={}".format(var, step))
    for i in range(len(sim_vars[var])):
        keys, values = zip(*sim_vars[var][i].items())
        ax[i].plot(keys, values)
    plt.show()
