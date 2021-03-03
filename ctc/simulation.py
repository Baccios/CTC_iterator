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

# from ctc.block_generator import get_ctc_assisted_circuit
from ctc.gates.cloning import CloningGate
from ctc.gates.ctc_assisted import CTCGate


class CTCCircuitSimulator:
    """
    This class provides tools to generate, run and visualize results of a
    simulation of a CTC assisted iterated circuit.
    """

    def __init__(self, size, k_value, cloning_size=7, base_block=None):
        """
        Initialize the simulator with a set of parameters.

        :param size: The number of bits used to encode k
        :type size: int
        :param k_value: The value for k, must be lower than 2^size
        :type k_value: int
        :param base_block: The gate to be used as basic block. When not specified,
                the class will use a default one using ctc.block_generator.get_ctc_assisted_gate
        :type base_block: qiskit.circuit.Gate
        :param cloning_size: the size of the internal CloningGate
        :type cloning_size: int
        """

        if size <= 0:
            raise ValueError("parameter size must be greater than zero")
        if k_value < 0 or k_value > (2 ** size - 1):
            raise ValueError("parameter k_value must be between zero and 2^size - 1")

        self._size = size
        self._k_value = k_value
        self._cloning_size = cloning_size
        if base_block is None:
            self._ctc_gate = CTCGate(2*size, method="v_gate", label="V Gate")
        else:
            if not isinstance(base_block, Gate):
                raise TypeError("parameter base_block must be a Gate")
            self._ctc_gate = base_block

    def _build_dctr(self, iterations, cloning="no_cloning"):
        """
        Build the dctr circuit using instance initialization parameters (utility)
        :param iterations: The number of iterations to build
        :type iterations: int
        :param cloning: can assume one of these values:
                    <ol>
                        <li>"no_cloning": Do not use cloning to replicate psi state</li>
                        <li>"initial": use cloning only for the first cloning_size iterations</li>
                        <li>"full": use cloning for each iteration
                    </ol>
        :type cloning: str
        :return: the full circuit ready to run.
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        init_gate = self._get_psi_init()

        qubits = QuantumRegister(2 * size)
        clone_qubits = QuantumRegister(self._cloning_size)
        dctr_circuit = QuantumCircuit(qubits)

        if cloning == "no_cloning":
            # initialize the first of CTC qubits with psi
            dctr_circuit.append(init_gate, [qubits[size]])
        else:
            dctr_circuit.add_register(clone_qubits)
            # initialize the cloning circuit
            dctr_circuit.append(init_gate, [clone_qubits[0]])
            dctr_circuit.append(CloningGate(self._cloning_size), clone_qubits)
            # use the first clone to initialize psi
            dctr_circuit.swap(qubits[size], clone_qubits[1])

        # useful to update next_clone_index
        def update_index(i):
            res = (i + 1) % self._cloning_size
            # print("Next clone index is: " + str(res))  # DEBUG
            return res

        next_clone_index = update_index(1)  # used to keep track of the clone to be used

        # place the initial CTC gate
        dctr_circuit.append(self._ctc_gate, qubits)

        # behavior depends on the value of cloning attribute:
        if cloning == "no_cloning":
            # in this case we simply apply default iterations with the real state
            iteration_instruction = self._get_iteration()
            for _ in range(iterations - 1):
                dctr_circuit.append(iteration_instruction, qubits)

        elif cloning == "initial":
            # in this case we use clones as long as they are available (self._cloning_size - 1)
            iteration_instruction = self._get_iteration()
            for _ in range(iterations - 1):
                if next_clone_index == 0:  # if clones are finished we use the real state
                    dctr_circuit.append(iteration_instruction, qubits)
                else:
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[next_clone_index]),
                        qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = update_index(next_clone_index)

        elif cloning == "full":
            # in this case we always use clones. If they are finished, we reset and clone again
            for _ in range(iterations - 1):
                if next_clone_index == 0:  # in this case we need to re init clones
                    for qubit in clone_qubits:
                        dctr_circuit.reset(qubit)
                    dctr_circuit.append(init_gate, [clone_qubits[0]])
                    dctr_circuit.append(CloningGate(self._cloning_size), clone_qubits)
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[1]), qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = 2
                else:
                    dctr_circuit.append(
                        self._get_iteration(psi_qubit=clone_qubits[next_clone_index]),
                        qubits[:] + clone_qubits[:]
                    )
                    next_clone_index = update_index(next_clone_index)

        else:
            raise ValueError("cloning must be either \"no_cloning\", \"initial\" or \"full\"")

        # dctr_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

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

    def _get_iteration(self, psi_qubit=None):
        """
        Get a single iteration of the circuit as a gate

        :param psi_qubit: The qubit containing psi to use as input to the CTC gate.
                             If set to None, psi is initialized using Initialize()
        :type psi_qubit: qiskit.circuit.Qubit
        :return: The iteration gate
        :rtype: qiskit.circuit.Instruction
        """
        size = self._size

        qubits = QuantumRegister(2 * size)

        iter_circuit = QuantumCircuit(qubits)

        if psi_qubit is not None:
            iter_circuit.add_register(psi_qubit.register)

        init_gate = self._get_psi_init()

        for i in range(size):
            iter_circuit.swap(qubits[i], qubits[size + i])

        iter_circuit.barrier()

        # reset CTC qubits
        for i in range(size, 2 * size):
            iter_circuit.reset(i)

        # initialize the first CTC gate
        if psi_qubit is None:
            iter_circuit.append(init_gate, [qubits[size]])
        else:
            iter_circuit.swap(qubits[size], psi_qubit)
        iter_circuit.append(self._ctc_gate, qubits)

        # iter_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

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

    def simulate(self, c_value, iterations, **params):
        """
        Simulate a run of the CTC iterated circuit

        :param c_value: The value for ancillary qubits. Must be between 0 and 2^size - 1
        :type c_value: int, str
        :param iterations: the number of iterations to simulate
        :type iterations: int
        :param params:
            List of accepted parameters:
            <ul>
              <li>
                cloning: can assume one of these values:
                <ol>
                    <li>"no_cloning": Do not use cloning to replicate psi state</li>
                    <li>"initial": use cloning only for the first cloning_size iterations</li>
                    <li>"full": use cloning for each iteration
                </ol>
              </li>
              <li>
                backend: The backend to use, defaults to QasmSimulator()
              </li>
              <li>
                shots: The number of shots for the simulation. Defaults to 512
              </li>
            </ul>
        :return: The count list of results
        :rtype: dict
        """
        backend = params.get("backend", QasmSimulator())
        cloning = params.get("cloning", "no_cloning")
        shots = params.get("shots", 512)

        dctr_simulation_circuit = self._build_simulator_circuit(c_value, iterations, cloning)

        job = execute(dctr_simulation_circuit, backend, shots=shots)

        # job_monitor(job) # DEBUG

        counts = job.result().get_counts()

        # plot_histogram(counts)  # DEBUG
        # plot.show()

        return counts

    def _build_simulator_circuit(self, c_value, iterations,
                                 cloning="no_cloning", add_measurements=True):
        """
        Build a dctr QuantumCircuit ready for a simulation (utility)
        :param c_value: the initial value for ancillary qubits
        :param iterations: the number of iterations in the circuit
        :param cloning: can assume one of these values:
                    <ol>
                        <li>"no_cloning": Do not use cloning to replicate psi state</li>
                        <li>"initial": use cloning only for the first cloning_size iterations</li>
                        <li>"full": use cloning for each iteration
                    </ol>
        :type cloning: str
        :param add_measurements: if True, also adds measurements at the end of the circuit
        :type add_measurements: bool
        :return: The circuit ready to be executed
        :rtype: qiskit.circuit.QuantumCircuit
        """

        if iterations <= 0:
            raise ValueError("parameter iterations must be greater than zero")

        dctr_instr = self._build_dctr(iterations, cloning)

        # initialize the final circuit
        classical_bits = ClassicalRegister(self._size)
        qubits = QuantumRegister(self._size * 2)

        dctr_simulation_circuit = QuantumCircuit(qubits, classical_bits)

        if cloning != "no_cloning":
            clone_qubits = QuantumRegister(self._cloning_size)
            dctr_simulation_circuit.add_register(clone_qubits)
            # initialize ancillary qubits
            self._initialize_c_state(c_value, dctr_simulation_circuit)
            dctr_simulation_circuit.append(dctr_instr, qubits[:] + clone_qubits[:])

        else:
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

    def test_convergence(self, c_value, start, stop, step=2, **params):
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
        :param params:
            List of accepted parameters:
            <ul>
              <li>
                cloning: can assume one of these values:
                <ol>
                    <li>"no_cloning": Do not use cloning to replicate psi state</li>
                    <li>"initial": use cloning only for the first cloning_size iterations</li>
                    <li>"full": use cloning for each iteration
                </ol>
              </li>
              <li>
                backend: The backend to use, defaults to QasmSimulator()
              </li>
              <li>
                shots: The number of shots for the simulation. Defaults to 1024
              </li>
            </ul>
        :return: None
        """
        cloning = params.get("cloning", "no_cloning")
        backend = params.get("backend", QasmSimulator())
        shots = params.get("shots", 1024)

        iterations = list(range(start, stop + 1, step))

        # if the backend is an IBMQ,
        # we submit all circuits together to optimize waiting time using IBMQJobManager
        if isinstance(backend, IBMQBackend):

            print("Building the circuits to submit...")
            circuits = \
                [self._build_simulator_circuit(c_value, i, cloning=cloning) for i in iterations]

            # Need to transpile the circuits first for optimization
            circuits = transpile(circuits, backend=backend)
            print("Circuits built and transpiled!")
            # circuits[len(circuits) - 1].draw(output="mpl", filename="./test.png")  # DEBUG

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
                count = self.simulate(c_value, i, cloning=cloning, backend=backend, shots=shots)
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
