# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

from qiskit import IBMQ
# from qiskit.visualization import plot_histogram

from ctc.simulation import CTCCircuitSimulator
from math import sqrt

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # provider = IBMQ.load_account()
    # backend = provider.get_backend('ibmq_santiago')
    """
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = least_busy(provider.backends(
                                        filters=lambda x: x.configuration().n_qubits >= 4
                                        and not x.configuration().simulator
                                        and x.status().operational is True
                                        and 'reset' in x.configuration().supported_instructions))
    print("least busy backend: ", backend)
    """

    c_values = ["0000", "0001", "0010", "0011", "0100", "0101", "0110", "0111", "1000", "1001", "1010",
                "1011", "1100", "1101", "1110", "1111"]

    c_values_cbs_vs_h = ["0001", [[1/sqrt(2), 1/sqrt(2)] for _ in range(4)],
                         [[1., 0]] + [[1 / sqrt(2), 1 / sqrt(2)] for _ in range(3)],
                         [[1., 0], [1., 0]] + [[1 / sqrt(2), 1 / sqrt(2)] for _ in range(2)],
                         [[1, 0], [1, 0], [1, 0], [1 / sqrt(2), 1 / sqrt(2)]],
                         "1000"]

    c_tick_labels = ["|" + str(i) + "⟩" for i in range(0, 16, 1)]

    c_tics_cbs_vs_h = ["|0001⟩", "|++++⟩", "|0+++⟩", "|00++⟩", "|000+⟩", "|1000⟩"]

    sim = CTCCircuitSimulator(size=4, k_value=0)
    sim.test_c_variability(c_values_cbs_vs_h, 1, 101, 20, c_tick_labels=c_tics_cbs_vs_h)
    """
    sim.test_convergence(c_value=c_value, start=1, stop=11, step=2, cloning="no_cloning")
    sim = CTCCircuitSimulator(size=2, k_value=2)
    counts = sim.simulate(c_value=1, iterations=6, cloning="no_cloning")

    plot_histogram(counts)
    plt.show()
    """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
