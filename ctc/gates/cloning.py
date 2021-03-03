"""
This module implements a deterministic quantum cloning gate
"""
from math import pi, sqrt

import numpy as np

# import matplotlib.pyplot as plt

from qiskit import QuantumRegister
from qiskit.circuit import Gate


class CloningGate(Gate):
    """
    Unitary, deterministic quantum cloning gate.
    """

    @staticmethod
    def _build_prep_gate():

        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        # creates the preparation gate
        prep_qr = QuantumRegister(2)
        prep_circuit = QuantumCircuit(prep_qr, name='prep')  # Create a quantum circuit with 2 qubits

        # Define the rotation angles
        theta1 = pi / 4.0
        temp2 = (3.0 - 2.0 * sqrt(2)) / 6.0
        arg2 = sqrt(temp2)
        theta2 = - 2.0 * np.arcsin(arg2)
        # print("THETA2 = ", theta2) # DEBUG
        theta3 = pi / 4.0

        prep_circuit.u(theta1, 0, 0, prep_qr[0])
        prep_circuit.cnot(prep_qr[0], prep_qr[1])
        prep_circuit.u(theta2, 0, 0, prep_qr[1])
        prep_circuit.cnot(prep_qr[1], prep_qr[0])
        prep_circuit.u(theta3, 0, 0, prep_qr[0])

        # Convert to a gate and stick it into an arbitrary place in the bigger circuit
        return prep_circuit.to_gate()

    def __init__(self, num_qubits, label=None):

        if (num_qubits % 2) != 1:
            raise ValueError("num_qubits must be odd.")

        self._num_qubits = num_qubits

        super().__init__('cloning', num_qubits, [], label)

    def _define(self):
        """
        define gate behavior
        """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit

        self._prep_gate = self._build_prep_gate()

        num_qubits = self._num_qubits

        qubits = QuantumRegister(num_qubits, "clone_q")
        cloning_circuit = QuantumCircuit(qubits)

        for i in range(1, num_qubits, 2):
            cloning_circuit.append(self._prep_gate, [qubits[i], qubits[i + 1]])

        # CNOT between the first qubit (control) and the odd qubits (targets)
        for i in range(1, num_qubits, 2):
            cloning_circuit.cnot(qubits[0], qubits[i])

        # CNOT between the first qubit (control) and the even qubits (targets)
        for i in range(2, num_qubits, 2):
            cloning_circuit.cnot(qubits[0], qubits[i])

        cloning_circuit.barrier()

        # CNOT between the first qubit (target) and the odd qubits (control)
        for i in range(1, num_qubits, 2):
            cloning_circuit.cnot(qubits[i], qubits[0])

        # CNOT between the first qubit (target) and the even qubits (control)
        for i in range(2, num_qubits, 2):
            cloning_circuit.cnot(qubits[i], qubits[0])

        # cloning_circuit.draw(output="mpl")  # DEBUG
        # plt.show()

        self.definition = cloning_circuit
