"""
This module implements the utilities required for the recipe in
<a href="https://arxiv.org/abs/0811.1209">this article</a> by Brun et al.
"""

import math
from math import pi
import numpy as np


def get_psi_statevector(k_value, num_bits):
    """
    Returns an array containing the statevector representation of |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>

    :param k_value: The value of the parameter k
    :param num_bits: The number of bits on which k is encoded
    :return: The numpy array containing the statevector representation of |psi>
    """
    angle = pi * k_value / 2 ** num_bits
    if k_value == 2 ** (num_bits - 1):  # optimization
        psi_vector = np.array([0., 1.])
    else:
        psi_vector = np.array([math.cos(angle), math.sin(angle)])
    return psi_vector


def get_psi_with_ancillas(psi_vector, num_qubits):
    """
    Appends to the statevector psi a number num_qubits - 1 of ancillary qubit states all set to |0>.

    :param psi_vector: The numpy array containing psi state in statevector representation.
    :param num_qubits: The total number of qubits the final state vector is composed of.
    :return: A numpy array containing the statevector representation of the state |psi>|0>|0>...|0>
    """
    ancillas_state = np.zeros(2**(num_qubits - 1))
    ancillas_state[0] = 1.
    psi_with_ancillas = np.tensordot(psi_vector, ancillas_state, axes = 0).reshape(1, 2**num_qubits)

    # print(psi_with_ancillas) # DEBUG
    return psi_with_ancillas


def create_set(num_bits):
    """
    Given the encoding |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>, Returns a list of state vectors of all possible
    values for k, numerically ordered. Each state is already filled with ancillary qubits all set to |0>.

    :param num_bits: The number of bits on which k is encoded.
    :return: A List of numpy arrays containing the state vectors of all possible values |psi_k>|0>...|0>,
    numerically ordered.
    """
    set_list = []
    for k in range(0, 2**num_bits):
        psi = get_psi_statevector(k, num_bits)
        psi_anc = get_psi_with_ancillas(psi, num_bits)
        set_list.append(psi_anc)

    return set_list


def create_cbs(num_qubits):
    """
    Returns the list of the CBS statevectors as numpy arrays.

    :param num_qubits: The number of qubits composing the CBS
    :return: The list containing CBS statevectors [|0>, |1>, ... , |2^num_qubits - 1>]
    """
    cbs_list = []
    for k in range(0, 2**num_qubits):
        psi = np.zeros(2 ** num_qubits)
        psi[k] = 1.
        cbs_list.append(psi)

    return cbs_list


def _get_b1_b2(k, num_bits, set_list):
    """
    Computes |b1> and |b2> using Grand-Schmidt orthogonalization.

    :param k: The index of U_k matrix.
    :param num_bits: The number of bits on which k is encoded.
    :param set_list: The set of 2^num_bits non-orthogonal quantum states.
    :return: A list containing the statevectors [|b1>, |b2>].
    """
    b1_vector = set_list[k]
    psi_2 = set_list[(k + 1) % 2**num_bits]
    dot_prod = np.dot(b1_vector, psi_2.reshape(2**num_bits, 1))
    b2_vector = psi_2 - dot_prod*b1_vector

    norm = np.linalg.norm(b2_vector)
    b2_vector = b2_vector/norm  # normalization
    return [b1_vector, b2_vector]


def _get_c1_c2(k, num_bits):
    """
    Computes |b1> and |b2> following Brun et al.'s recipe.

    :param k: The index of U_k matrix.
    :param num_bits: The number of bits on which k is encoded.
    :return: A list containing the statevectors [|c1>, |c2>].
    """

    c1_vector = np.zeros(2**num_bits)
    c1_vector[k] = 1.

    cbs_set = create_cbs(num_bits)
    cbs_set.pop(k)
    norm_factor = 1/math.sqrt(2**num_bits - 1)
    c2_vector = np.zeros(2**num_bits)
    for psi in cbs_set:
        c2_vector += psi
    c2_vector *= norm_factor

    return [c1_vector, c2_vector]


def _fill_basis_utility(basis, count, target, num_bits):
    if count == target:
        return

    # print("A = ", np.vstack(basis))  # DEBUG

    # find linear independent vectors
    # at least one CBS vector is linearly independent
    for e in create_cbs(num_bits):
        # print()
        # print("Trying candidate e = ", e)
        # print()
        # try to orthogonalize the CBS state
        column_e = e.copy().reshape(2 ** num_bits, 1)
        candidate_e = e.reshape(1, 2**num_bits)
        for psi in basis:
            factor = np.dot(psi, column_e)
            candidate_e -= factor * psi
        norm = np.linalg.norm(candidate_e)
        # print("Norm for this candidate is ", norm)
        if norm > 0.000000001:  # in this case e was linearly independent and has been orthogonalized
            # print("Candidate accepted! Vector added = ", candidate_e/norm)
            new_vector = candidate_e/norm
            basis.append(new_vector)


def _fill_basis(basis, num_bits):
    """
    Given an incomplete orthonormal set of vectors, Completes the set to form an orthonormal basis
    :param basis: The incomplete basis to be filled.
    :param num_bits: The number of bits on which k is encoded
    :return: None
    """
    _fill_basis_utility(basis, len(basis), 2**num_bits, num_bits)
    # print(np.vstack(basis))  # DEBUG
    return np.vstack(basis)


def get_u(k, num_bits):
    """
    Compute U_k matrix as specified by Brun et al. in <li>"brun": use the algorithm in
    <a href="https://arxiv.org/abs/0811.1209">this article</a>

    :param k: The index of matrix U_k
    :param num_bits: The number of bits on which k is encoded.
    :return: The U matrix as a numpy 2-dimensional array.
    """

    states_set = create_set(num_bits)
    b_basis = _get_b1_b2(k, num_bits, states_set)
    c_basis = _get_c1_c2(k, num_bits)
    _fill_basis(b_basis, num_bits)
    _fill_basis(c_basis, num_bits)

    u_matrix = np.zeros(2**(2*num_bits))
    u_matrix = u_matrix.reshape(2**num_bits, 2**num_bits)
    for i in range(len(b_basis)):
        u_matrix += np.dot(c_basis[i].reshape(2**num_bits, 1), b_basis[i])
    # print("U = ", u_matrix)  # DEBUG

    return u_matrix


def get_u_qiskit(k, num_bits):
    """
    See References for get_u, the only difference is that the matrix returned by this method can be used on the
    reverse ordered qubits in Qiskit
    """

    swapper = _get_qiskit_inverter(num_bits)
    u = get_u(k, num_bits)
    u = np.dot(np.dot(swapper, u), swapper)
    return u


def _get_qiskit_inverter(num_bits):
    """
    Since Qiskit considers qubits in reverse order, unitaries must be adapted to Qiskit by pre- and post-multiplying
    them by the matrix returned by this function
    :param num_bits: The size of the matrix is 2**num_bits
    :return: The matrix to pre- and post-multiply to the given unitary.
    """

    size = 2**num_bits
    matrix = np.zeros(size*size)
    matrix = matrix.reshape(size, size)

    for i in range(size):
        reverse = format(i, '0' + str(num_bits) + 'b')[::-1]
        # print("i is ", i, ", rev is ", reverse)  # DEBUG
        reverse = int(reverse, 2)  # DEBUG
        # print("reverse is now ", reverse)  # DEBUG

        matrix[i][reverse] = 1

    return matrix

