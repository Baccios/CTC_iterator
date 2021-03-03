"""
This module provides an API to generate parametrized CTC assisted circuits.
With CTC circuit we mean a gate respecting
Deutschian Closed Timelike Curve definition
"""
from math import pi

from qiskit import QuantumCircuit


def _generate_v_circuit(size):
    """
    Get a CTC gate using the algorithm in
    <a href="https://arxiv.org/abs/1901.00379">this article</a>

    :param size: Size (in qubits) of the gate instance
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # build the V sub circuit
    ctc_circuit = QuantumCircuit(2 * size, name='V Gate')

    # ### R block

    for i in range(size):
        ctc_circuit.cu(-pi / 2 ** i, 0, 0, 0, i, size)

    # ### T block

    for i in range(size + 1, 2 * size):
        ctc_circuit.ch(size, i)

    # ### W block

    for i in range(size + 1, 2 * size):
        ctc_circuit.cu(pi / size, 0, 0, 0, i, size)

    # ### C block

    for i in range(size):
        ctc_circuit.cnot(i, size + i)

    # return the result
    return ctc_circuit


def get_ctc_assisted_circuit(size, method="v_gate"):
    """
    Get a CTC gate specifying its size and (optionally) the method used to build it.

    :param size: The size (in qubits) of the gate.
                 The resulting gate will have 2*size qubits because
                 the first half represents the CTC
                 and the second half represents the Chronology Respecting (CR) system.
    :type size: int
    :param method: the algorithm used to build the gate. It defaults to "v_gate".
                   Possible values are:
                    <ol>
                        <li>"v_gate": use the algorithm in
                            <a href="https://arxiv.org/abs/1901.00379">this article</a>
                        </li>
                    </ol>
    :type method: str
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # Other methods are left for future updates
    if method != "v_gate":
        raise ValueError("method must be set to one of the specified values")
    return _generate_v_circuit(size)
