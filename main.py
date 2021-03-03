# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import matplotlib.pyplot as plt

from qiskit import IBMQ
from qiskit.visualization import plot_histogram

from ctc.simulation import CTCCircuitSimulator

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    provider = IBMQ.load_account()
    backend = provider.get_backend('ibmq_santiago')
    """
    provider = IBMQ.get_provider(hub='ibm-q', group='open', project='main')
    backend = least_busy(provider.backends(
                                        filters=lambda x: x.configuration().n_qubits >= 4
                                        and not x.configuration().simulator
                                        and x.status().operational is True
                                        and 'reset' in x.configuration().supported_instructions))
    print("least busy backend: ", backend)
    """

    sim = CTCCircuitSimulator(size=2, k_value=2, cloning_size=7)
    sim.test_convergence(c_value=0, start=1, stop=21, step=2, cloning="full")
    """
    sim = CTCCircuitSimulator(size=2, k_value=2)
    counts = sim.simulate(c_value=1, iterations=6, cloning="no_cloning")

    plot_histogram(counts)
    plt.show()
    """

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
