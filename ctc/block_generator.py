"""
This module provides an API to generate parametrized CTC assisted circuits.
With CTC circuit we mean a gate used for an interaction with a
Deutschian Closed Timelike Curve.
"""
from math import pi

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.extensions import UnitaryGate

from ctc.brun import get_u_qiskit


def _generate_v_circuit_nbp(size):
    """
    Get a CTC gate using the recipe in
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


def _generate_v_circuit_brun_fig2(size=2):
    """
    Get a CTC gate using the recipe in Fig. 2 of
    <a href="https://arxiv.org/abs/0811.1209">this article</a>

    :param size: Size (in qubits) of the gate instance (only value of 2 is implemented for this method)
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """

    if size != 2:
        raise ValueError("Error. This algorithm has only been implemented with a size n=2. Use \"brun\" instead.")

    u00 = QuantumCircuit(2, name="U00")
    u00.swap(0, 1)

    u01 = QuantumCircuit(2, name="U01")
    u01.x(0)
    u01.x(1)

    u10 = QuantumCircuit(2, name="U10")
    u10.h(0)
    u10.x(0)

    u11 = QuantumCircuit(2, name="U11")
    u11.swap(0, 1)
    u11.x(0)
    u11.h(1)

    cc_u00 = u00.control(2, ctrl_state='00')
    cc_u01 = u01.control(2, ctrl_state='10')
    cc_u10 = u10.control(2, ctrl_state='01')
    cc_u11 = u11.control(2, ctrl_state='11')

    qr = QuantumRegister(4)
    ctc_circuit = QuantumCircuit(qr, name="V_gate")
    ctc_circuit.append(cc_u00.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u01.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u10.to_instruction(), qr[0:4])
    ctc_circuit.append(cc_u11.to_instruction(), qr[0:4])

    # return the result
    return ctc_circuit


def _generate_v_circuit_brun(size):
    """
    Get a CTC gate using the general recipe in
    <a href="https://arxiv.org/abs/0811.1209">this article</a>

    :param size: Size (in qubits) of the gate instance
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """

    def get_str(k):
        return format(k, '0' + str(size) + 'b')

    qr = QuantumRegister(2*size)
    ctc_circuit = QuantumCircuit(qr, name="V_gate")

    for k in range(2**size):
        u_k = UnitaryGate(get_u_qiskit(k, size), label="U_" + get_str(k))
        cu_k = u_k.control(size, label="CU_" + get_str(k), ctrl_state=get_str(k)[::-1])
        # print("Straight ", k, " = ", get_str(k))
        # print("Reverse ", k, " = ", get_str(k)[::-1])  # DEBUG
        ctc_circuit.append(cu_k, qr[0:2**size])

    return ctc_circuit


def _generate_v_circuit_momentum(size):
    """
    Get a CTC gate using the algorithm in
    <a href="https://arxiv.org/abs/1901.00379">this article</a>

    :param size: Size (in qubits) of the gate instance
    :type size: int
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # build the V sub circuit
    ctc_circuit = QuantumCircuit(3 * size, name='MV Gate')

    # ### R block (1)

    for i in range(size):
        ctc_circuit.cu(-pi / 2 ** (i+1), 0, 0, 0, i, 2*size)

    # ### R block (2)

    for i in range(size):
        ctc_circuit.cu(-pi / 2 ** (i+1), 0, 0, 0, size + i, 2*size)

    # ### T block

    for i in range(2*size + 1, 3 * size):
        ctc_circuit.ch(2*size, i)

    # ### W block

    for i in range(2*size + 1, 3 * size):
        ctc_circuit.cu(pi / size, 0, 0, 0, i, 2*size)

    # ### C block

    for i in range(size):
        ctc_circuit.cnot(size + i, 2*size + i)

    # return the result
    return ctc_circuit


def get_ctc_assisted_circuit(size, method="nbp"):
    """
    Get a CTC gate specifying its size and (optionally) the method used to build it.

    :param size: The size (in qubits) of the gate.
                 The resulting gate will have 2*size qubits because
                 the first half represents the CTC
                 and the second half represents the Chronology Respecting (CR) system.
    :type size: int
    :param method: the algorithm used to build the gate. It defaults to "nbp".
                   Possible values are:
                    <ol>
                        <li>"nbp": use the algorithm in
                            <a href="https://arxiv.org/abs/1901.00379">this article</a>
                        </li>
                    </ol>
    :type method: str
    :return: the CTC gate
    :rtype: qiskit.circuit.QuantumCircuit
    """
    # Other methods are left for future updates
    if method == "nbp":
        return _generate_v_circuit_nbp(size)
    elif method == "brun_fig2":
        return _generate_v_circuit_brun_fig2(size)
    elif method == "brun":
        return _generate_v_circuit_brun(size)
    else:
        raise ValueError("method must be set to one of the specified values")


def get_ctc_assisted_circuit_mom(size, method="nbp"):
    """
    Get a Momentum CTC gate specifying its size and (optionally) the method used to build it.

    :param size: The size (in qubits) of the gate.
                 The resulting gate will have 3*size qubits because
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
    if method != "nbp":
        raise ValueError("Momentum variation has been implemented only for nbp method.")
    return _generate_v_circuit_momentum(size)
