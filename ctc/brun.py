"""
This module implements the utilities required for the recipe in
<a href="https://arxiv.org/abs/0811.1209">this article</a> by Brun et al.
"""

import math
import numpy as np

from ctc.encodings import get_2d_code, get_3d_code, get_psi_with_ancillas


def create_set(num_bits, two_dim=True, section_divider=None):
    """
    Returns a list of state vectors of all possible
    values for k, numerically ordered. Each state is already filled with ancillary qubits all set to |0>.

    :param num_bits: The number of bits on which k is encoded.
    :param two_dim: if set to True, the encoding considered is |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>, otherwise
    it is the 3d encoding scheme. Defaults to True.
    :param section_divider: If two_dim is True, the statevector representation will be
    |psi_k> = cos(k*pi/sector_divider)|0> + sin(k*pi/sector_divider)|1>. If None, it will be considered as 2^size.
    :return: A List of numpy arrays containing the state vectors of all possible values |psi_k>|0>...|0>,
    numerically ordered.
    """
    set_list = []
    for k in range(0, 2**num_bits):
        if two_dim:
            psi = get_2d_code(k, num_bits, section_divider)
        else:
            psi = get_3d_code(k, num_bits)
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
        psi = np.zeros(2 ** num_qubits, dtype=complex)
        psi[k] = complex(real=1)
        cbs_list.append(psi)

    return cbs_list


def _get_b1_b2(k, num_bits, set_list):
    """
    Computes |b1> and |b2> using Gram-Schmidt orthogonalization.

    :param k: The index of U_k matrix.
    :param num_bits: The number of bits on which k is encoded.
    :param set_list: The set of 2^num_bits non-orthogonal quantum states.
    :return: A list containing the statevectors [|b1>, |b2>].
    """
    b1_vector = set_list[k]
    psi_2 = set_list[(k + 1) % 2**num_bits]
    dot_prod = np.dot(b1_vector.conjugate(), psi_2.reshape(2**num_bits, 1))
    b2_vector = psi_2 - dot_prod*b1_vector

    norm = np.linalg.norm(b2_vector)
    b2_vector = b2_vector/norm  # normalization
    return [b1_vector, b2_vector]


def _get_c1_c2(k, num_bits):
    """
    Computes |c1> and |c2> following Brun et al.'s recipe.

    :param k: The index of U_k matrix.
    :param num_bits: The number of bits on which k is encoded.
    :return: A list containing the statevectors [|c1>, |c2>].
    """

    c1_vector = np.zeros(2**num_bits, dtype=complex)
    c1_vector[k] = complex(real=1.)

    cbs_set = create_cbs(num_bits)
    cbs_set.pop(k)
    norm_factor = 1/math.sqrt(2**num_bits - 1)
    c2_vector = np.zeros(2**num_bits, dtype=complex)
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
        # print()  # DEBUG
        # try to orthogonalize the CBS state
        column_e = e.copy().reshape(2 ** num_bits, 1)
        candidate_e = e.reshape(1, 2**num_bits)
        for psi in basis:
            factor = np.dot(psi.conjugate(), column_e)
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


def get_u_old(k, num_bits, two_dim=True, section_divider=None, phi=None):
    """
    Compute U_k matrix as specified by Brun et al. in
    <a href="https://arxiv.org/abs/0811.1209">this article</a>.

    :param k: The index of matrix U_k
    :param num_bits: The number of bits on which k is encoded.
    :param two_dim: if set to True, the encoding considered is |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>, otherwise
    it is the 3d encoding scheme. Defaults to True.
    :param section_divider: If two_dim is True and section_divider is not None, the statevector representation will be
    |psi_k> = cos(k*pi/sector_divider)|0> + sin(k*pi/sector_divider)|1>.
    :return: The U matrix as a numpy 2-dimensional array.
    """

    states_set = create_set(num_bits, two_dim, section_divider)

    psi_k_ancillas = states_set[k][0]
    psi_k = np.array([psi_k_ancillas[0], psi_k_ancillas[2**(num_bits - 1)]])
    k_cbs_state = np.zeros(2**num_bits, dtype=complex)
    k_cbs_state[k] = 1.

    k_xor_1 = (k + 1) % 2**num_bits

    k_xor_1_state = np.zeros(2**num_bits, dtype=complex)
    k_xor_1_state[k_xor_1] = 1.

    psi_k_ort_ancillas = states_set[(k + 1) % 2**num_bits][0]
    psi_k_ort = np.array([psi_k_ort_ancillas[0], psi_k_ort_ancillas[2**(num_bits - 1)]])
    column_psi_k_ort = np.array(psi_k_ort, copy=True).reshape(2, 1)
    factor = np.dot(psi_k.conjugate(), column_psi_k_ort)
    psi_k_ort -= factor * psi_k

    psi_k_ort /= np.linalg.norm(psi_k_ort)

    # reshape as a column vector
    k_cbs_state = k_cbs_state.reshape(2**num_bits, 1)
    k_xor_1_state = k_xor_1_state.reshape(2**num_bits, 1)

    v_matrix = np.dot(k_cbs_state, psi_k.conjugate().reshape(1,2))
    v_matrix += np.dot(k_xor_1_state, psi_k_ort.conjugate().reshape(1,2))

    # print(v_matrix)

    u_matrix = np.zeros(shape=(2**num_bits, 2**num_bits), dtype=complex)
    u_matrix[:, 0] = v_matrix[:, 0]
    u_matrix[:, 2**(num_bits-1)] = v_matrix[:, 1]

    j = 0
    for i in range(2**num_bits):
        if i != 0 and i != 2**(num_bits - 1):
            while j == k or j == k_xor_1:
                j += 1
            u_matrix[j, i] = 1.
            j += 1

    # print("U_", k, "= ", u_matrix)  # DEBUG
    # print("I = ", np.dot(u_matrix, np.matrix.getH(u_matrix)))  # DEBUG

    return u_matrix


def get_u(k, num_bits, two_dim=True, section_divider=None):
    """
    Compute U_k matrix as specified by Brun et al. in
    <a href="https://arxiv.org/abs/0811.1209">this article</a>.

    :param k: The index of matrix U_k
    :param num_bits: The number of bits on which k is encoded.
    :param two_dim: if set to True, the encoding considered is |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>, otherwise
    it is the 3d encoding scheme. Defaults to True.
    :param section_divider: If two_dim is True and section_divider is not None, the statevector representation will be
    |psi_k> = cos(k*pi/sector_divider)|0> + sin(k*pi/sector_divider)|1>.
    :return: The U matrix as a numpy 2-dimensional array.
    """

    states_set = create_set(num_bits, two_dim, section_divider)
    b_basis = _get_b1_b2(k, num_bits, states_set)
    c_basis = _get_c1_c2(k, num_bits)

    _fill_basis(b_basis, num_bits)
    _fill_basis(c_basis, num_bits)

    u_matrix = np.zeros(2**(2*num_bits), dtype=complex)
    u_matrix = u_matrix.reshape(2**num_bits, 2**num_bits)
    for i in range(len(b_basis)):
        u_matrix += np.dot(c_basis[i].reshape(2**num_bits, 1), b_basis[i].conjugate())
    # print("U_", k, "= ", u_matrix)  # DEBUG
    # print("I = ", np.dot(u_matrix, np.matrix.getH(u_matrix)))  # DEBUG

    return u_matrix


def get_u_qiskit(k, num_bits, two_dim=True, section_divider=None):
    """
    See References for get_u, the only difference is that the matrix returned by this method can be used on Qiskit,
    which has reverse ordered qubits.
    """

    swapper = _get_qiskit_inverter(num_bits)
    u = get_u(k, num_bits, two_dim, section_divider)
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

