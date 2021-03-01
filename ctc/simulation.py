"""
This module provides an API to perform simulations of an iterated CTC cloning circuit.
"""

import math
import os
from math import pi, sqrt

import scipy.stats

import numpy as np

from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, execute, transpile
from qiskit.circuit import Gate
from qiskit.extensions import Initialize
from qiskit.providers.aer import QasmSimulator
from qiskit.providers.ibmq import IBMQBackend
from qiskit.tools.monitor import job_monitor
from qiskit.providers.ibmq.managed import IBMQJobManager

# import basic plot tools
import matplotlib.pyplot as plt

from ctc.block_generator import get_ctc_assisted_gate


class CTCCircuitSimulator:
    """
    This class provides tools to generate, run and visualize results of a
    simulation of a CTC assisted iterated circuit.
    """

    def __init__(self, size, k_value, base_block=None):
        """
        Initialize the simulator with a set of parameters.

        :param size: The number of bits used to encode k
        :type size: int
        :param k_value: The value for k, must be lower than 2^size
        :type k_value: int
        :param base_block: The gate to be used as basic block. When not specified,
                the class will use a default one using ctc.block_generator.get_ctc_assisted_gate
        :type base_block: qiskit.circuit.Gate
        """

        if size <= 0:
            raise ValueError("parameter size must be greater than zero")
        if k_value < 0 or k_value > (2 ** size - 1):
            raise ValueError("parameter k_value must be between zero and 2^size - 1")

        self._size = size
        self._k_value = k_value
        if base_block is None:
            self._ctc_gate = get_ctc_assisted_gate(size)
        else:
            if not isinstance(base_block, Gate):
                raise TypeError("parameter base_block must be a Gate")
            self._ctc_gate = base_block

    def _build_dctr(self, iterations):
        """
        Build the dctr circuit using instance initialization parameters (utility)
        :param iterations: The number of iterations to build
        :return: the full circuit ready to run.
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        init_gate = self._get_psi_init()

        qubits = QuantumRegister(2 * size)
        dctr_circuit = QuantumCircuit(qubits)

        # initialize the first of CTC qubits with psi
        dctr_circuit.append(init_gate, [qubits[size]])

        # place the initial CTC gate
        dctr_circuit.append(self._ctc_gate, qubits)

        iteration_instruction = self._get_iteration()

        for _ in range(iterations - 1):
            dctr_circuit.append(iteration_instruction, qubits)

        # dctr_circuit.draw(output="mpl", filename="./test.png")  # DEBUG

        return dctr_circuit.to_instruction()

    def _get_psi_init(self):
        """
        Get an Initialize gate for psi state
        :return: the gate used to initialize psi
        :rtype: qiskit.circuit.Gate
        """

        # encode k in a state |ψ⟩ = cos(kπ/2^n)|0⟩ + sin(kπ/2^n)|1⟩
        psi = [
            math.cos((self._k_value * pi) / 2 ** self._size),
            math.sin((self._k_value * pi) / 2 ** self._size)
        ]

        # print("k = ", self.__k_value, ", psi is encoded as: ", psi)  # DEBUG

        # Let's create our initialization gate to create |ψ⟩
        init_gate = Initialize(psi)
        init_gate.label = "psi-init"
        return init_gate

    def _get_iteration(self):
        """
        Get a single iteration of the circuit as a gate

        :return: The iteration gate
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        qubits = QuantumRegister(2 * size)
        iter_circuit = QuantumCircuit(qubits)

        init_gate = self._get_psi_init()

        for i in range(size):
            iter_circuit.swap(qubits[i], qubits[size + i])

        iter_circuit.barrier()

        # reset CTC qubits
        for i in range(size, 2 * size):
            iter_circuit.reset(i)

        # initialize the first CTC gate
        iter_circuit.append(init_gate, [qubits[size]])
        iter_circuit.append(self._ctc_gate, qubits)

        return iter_circuit.to_instruction()

    @staticmethod
    def _check_binary(c_value_string):
        """
        Check if a string contains only binary characters (utility)

        :param c_value_string: The string to check
        :return: True if the string is binary
        """
        set_value = set(c_value_string)
        if set_value in ({'0', '1'}, {'0'}, {'1'}):
            return True
        return False

    # takes in input c either as a binary string (little endian) (e.g. '1001')
    # or as an integer and initializes the circuit with such CBS state
    def _initialize_c_state(self, c_value, dctr_circuit):
        """
        Initialize ancillary qubits using a binary string or an integer.
        takes in input c either as a binary string (little endian) (e.g. '1001')
        or as an integer and initializes the circuit with such CBS state. (utility)

        :param c_value: the value to initialize
        :type c_value: str, int
        :param dctr_circuit: The quantum circuit to be initialized
        :type dctr_circuit: qiskit.circuit.quantumcircuit.QuantumCircuit
        :return: None
        """

        if isinstance(c_value, int):
            string_state = format(c_value, '0' + str(self._size) + 'b')
        elif isinstance(c_value, str):
            string_state = c_value
        else:
            raise TypeError("c value was not a string or an integer")

        if (len(string_state) != self._size) or not self._check_binary(string_state):
            raise ValueError("c string must be binary, e.g. \"10001\"")

        for i, bit in enumerate(string_state):
            if bit == '1':
                dctr_circuit.x(i)

    def simulate(self, c_value, iterations, backend=QasmSimulator(), shots=512):
        """
        Simulate a run of the CTC iterated circuit

        :param c_value: The value for ancillary qubits. Must be between 0 and 2^size - 1
        :type c_value: int, str
        :param iterations: the number of iterations to simulate
        :type iterations: int
        :param backend: The backend to use, defaults to QasmSimulator()
        :type backend: qiskit.providers.aer.backends.aerbackend.AerBackend
        :param shots: THe number of shots for the simulation
        :return: The count list of results
        :rtype: dict
        """
        dctr_simulation_circuit = self._build_simulator_circuit(c_value, iterations)

        job = execute(dctr_simulation_circuit, backend, shots=shots)

        job_monitor(job)

        counts = job.result().get_counts()

        # plot_histogram(counts)  # DEBUG
        # plot.show()

        return counts

    def _build_simulator_circuit(self, c_value, iterations, add_measurements=True):
        """
        Build a dctr QuantumCircuit ready for a simulation (utility)
        :param c_value: the initial value for ancillary qubits
        :param iterations: the number of iterations in the circuit
        :param add_measurements: if True, also adds measurements at the end of the circuit
        :type add_measurements: bool
        :return: The circuit ready to be executed
        :rtype: qiskit.circuit.QuantumCircuit
        """

        if iterations <= 0:
            raise ValueError("parameter iterations must be greater than zero")

        dctr_instr = self._build_dctr(iterations)

        # initialize the final circuit
        qubits = QuantumRegister(self._size * 2)
        classical_bits = ClassicalRegister(self._size)
        dctr_simulation_circuit = QuantumCircuit(qubits, classical_bits)

        # initialize ancillary qubits
        self._initialize_c_state(c_value, dctr_simulation_circuit)
        dctr_simulation_circuit.append(dctr_instr, qubits)

        if add_measurements:
            # noinspection PyTypeChecker
            self._add_measurement(dctr_simulation_circuit)

        return dctr_simulation_circuit

    def _add_measurement(self, dctr_circuit):
        classical_bits = dctr_circuit.clbits
        qubits = dctr_circuit.qubits
        for i in range(self._size):
            dctr_circuit.measure(qubits[i + self._size],
                                 classical_bits[self._size - 1 - i])

    def _binary(self, value):
        """
        Formats value as a binary string on size digits (utility)

        :param value: the integer to be formatted
        :return: the binary string
        """
        return ('{0:0' + str(self._size) + 'b}').format(value)

    def test_convergence(self, c_value, start, stop, step=2, backend=QasmSimulator(), shots=1024):
        """
        Test the convergence rate of the algorithm by simulating it
        under an increasing number of iterations. Save the output plots in ./out

        :param c_value: The value for the ancillary qubits, either as binary string or integer
        :type c_value: str, int
        :param start: the starting number of iterations
        :type start: int
        :param stop: the final number of iterations
        :type stop: int
        :param step: the iterations increase step
        :type step: int
        :param backend: The backend to use, defaults to QasmSimulator()
        :type backend: qiskit.aer.backends.aerbackend.AerBackend
        :param shots: The number of shots for each simulation, defaults to 1024
        :type shots: int
        :return: None
        """

        iterations = list(range(start, stop + 1, step))

        # if the backend is an IBMQ,
        # we submit all circuits together to optimize waiting time using IBMQJobManager
        if isinstance(backend, IBMQBackend):

            print("Building the circuits to submit...")
            circuits = [self._build_simulator_circuit(c_value, i) for i in iterations]

            # Need to transpile the circuits first for optimization
            circuits = transpile(circuits, backend=backend)
            print("Circuits built and transpiled!")

            # Use Job Manager to break the circuits into multiple jobs.
            job_manager = IBMQJobManager()
            job_set = job_manager.run(circuits, backend=backend, name='test_convergence')

            # we monitor the first job in the list to have some feedback
            job_monitor(job_set.jobs()[0])

            results = job_set.results()

            # extract normalized probabilities of success
            probabilities = [
                results.get_counts(i)[self._binary(self._k_value)] / shots
                for i in range(len(circuits))
            ]

            conf_intervals_95 = [
                scipy.stats.norm.ppf(0.975) * sqrt(probabilities[i]
                                                   * (1 - probabilities[i]) / shots)
                for i in range(len(circuits))
            ]

        else:
            probabilities = []
            conf_intervals_95 = []

            for i in iterations:
                count = self.simulate(c_value, i, backend=backend, shots=shots)
                norm_shots = sum(count.values())  # should be equal to shots
                success_prob = count[self._binary(self._k_value)] / norm_shots
                confidence_int_95 = scipy.stats.norm.ppf(0.975) * \
                    sqrt(success_prob * (1 - success_prob) / norm_shots)

                probabilities.append(success_prob)
                conf_intervals_95.append(confidence_int_95)

        # select the first plot
        plt.figure(1)

        plt.ylabel('Probability')
        plt.xlabel('Iterations')
        plt.title("Bar plot: n_qbits = " + str(self._size) +
                  ", initial_state I" + self._binary(c_value) + ">")

        # build the bar plot
        x_positions = np.arange(len(probabilities))
        plt.bar(
            x_positions,
            probabilities,
            color='blue',
            edgecolor='black',
            yerr=conf_intervals_95,
            capsize=7,
            label='success probability'
        )

        plt.xticks(x_positions, iterations)
        plt.grid(axis='y', linestyle='--', linewidth=0.5)

        if not os.path.exists('out'):
            try:
                os.makedirs('out')
            except OSError as _:
                print("Error: could not create \"./out\" directory")
                return

        # finally save the plot
        image_basename = 'out/' + str(self._size) + '_convergence'
        plt.savefig(image_basename + '_bar.png')

        # select the second plot
        plt.figure(2)

        plt.ylabel('Probability')
        plt.xlabel('Iterations')
        plt.title("Convergence log-log rate: n_qbits = " + str(self._size) +
                  ", initial_state I" + self._binary(c_value) + ">")

        # build log-log plot
        plt.errorbar(
            iterations, probabilities, fmt='o', yerr=conf_intervals_95, label='success_probability'
        )
        plt.loglog(iterations, probabilities, color='blue')
        plt.xscale('log', base=2)
        plt.yscale('log', base=2)
        plt.grid()

        # save also the second plot
        plt.savefig(image_basename + '_log.png')
