"""
This utility module implements some different encodings that are used in the simulations.
"""

import numpy as np
from math import pi
import math


def get_2d_code(k_value, num_bits, sector_divider=None, phi=None):
    """
    Returns an array containing the statevector representation of |psi_k> = cos(k*pi/2^n)|0> + sin(k*pi/2^n)|1>

    :param k_value: The value of the parameter k
    :param num_bits: The number of bits on which k is encoded
    :param sector_divider: if not None, the statevector representation will be
    |psi_k> = cos(k*pi/sector_divider)|0> + sin(k*pi/sector_divider)|1>. Defaults to None.
    :param phi: If not None, The statevector representation plane will be rotated horizontally
    by the angle phi in radiants.
    :type phi: float
    :return: The numpy array containing the statevector representation of |psi_k>
    """

    if sector_divider is None:
        sector_divider = 2**num_bits
    angle = pi * k_value / sector_divider
    if phi is not None:
        psi_vector = np.array([complex(real=math.cos(angle)), complex(real=math.sin(angle)*math.cos(phi),
                                                                      imag=math.sin(angle)*math.sin(phi))])
    else:
        psi_vector = np.array([complex(real=math.cos(angle)), complex(real=math.sin(angle))])
    return psi_vector


def get_3d_code(k_value, num_bits):
    """
    Return the statevector corresponding to k_value and num_bits according to the 3d encoding scheme.
    :param k_value: The integer representing the statevector. It must be between 0 and 2**num_bits - 1
    :param num_bits: The number of bits which k_value is composed of.
    :return: The statevector as a numpy array.
    """

    if num_bits == 2:
        angle = 0.304085*pi
        if k_value == 0:
            psi_vector = np.array([1., 0.])
        elif k_value == 1:
            psi_vector = np.array([math.cos(angle), math.sin(angle)])
        elif k_value == 2:
            psi_vector = np.array([math.cos(angle), math.sin(angle) * complex(math.cos(-2 * pi / 3), math.sin(-2 * pi / 3))])
        else:  # k_value == 3:
            psi_vector = np.array([math.cos(angle), math.sin(angle) * complex(math.cos(2 * pi / 3), math.sin(2 * pi / 3))])

    elif num_bits == 3:
        angle = 0.15205*pi
        angle_2 = 0.34795*pi
        if k_value == 0:
            psi_vector = np.array([math.cos(angle), math.sin(angle)])
        elif k_value == 1:
            psi_vector = np.array([math.cos(angle), math.sin(angle) * complex(0, 1)])
        elif k_value == 2:
            psi_vector = np.array([math.cos(angle), -math.sin(angle)])
        elif k_value == 3:
            psi_vector = np.array([math.cos(angle), math.sin(angle) * complex(0, -1)])
        elif k_value == 4:
            psi_vector = np.array([math.cos(angle_2), math.sin(angle_2)])
        elif k_value == 5:
            psi_vector = np.array([math.cos(angle_2), math.sin(angle_2) * complex(0, 1)])
        elif k_value == 6:
            psi_vector = np.array([math.cos(angle_2), -math.sin(angle_2)])
        else:  # k_value == 7:
            psi_vector = np.array([math.cos(angle_2), math.sin(angle_2) * complex(0, -1)])

    elif num_bits == 4:
        if k_value == 0:
            psi_vector = np.array([math.cos(pi/10), math.sin(pi/10)])
        elif k_value == 1:
            psi_vector = np.array([math.cos(2*pi/10), math.sin(2*pi/10)])
        elif k_value == 2:
            psi_vector = np.array([math.cos(3*pi/10), math.sin(3*pi/10)])
        elif k_value == 3:
            psi_vector = np.array([math.cos(4*pi/10), math.sin(4*pi/10)])
        elif k_value == 4:
            psi_vector = np.array([math.cos(pi/10), math.sin(pi/10) * complex(math.cos(pi/2), math.sin(pi/2))])
        elif k_value == 5:
            psi_vector = np.array([math.cos(2*pi/10), math.sin(2*pi/10) * complex(math.cos(pi/2), math.sin(pi/2))])
        elif k_value == 6:
            psi_vector = np.array([math.cos(3*pi/10), math.sin(3*pi/10) * complex(math.cos(pi/2), math.sin(pi/2))])
        elif k_value == 7:
            psi_vector = np.array([math.cos(4*pi/10), math.sin(4*pi/10) * complex(math.cos(pi/2), math.sin(pi/2))])
        elif k_value == 8:
            psi_vector = np.array([math.cos(pi/10), math.sin(pi/10) * complex(math.cos(pi), math.sin(pi))])
        elif k_value == 9:
            psi_vector = np.array([math.cos(2*pi/10), math.sin(2*pi/10) * complex(math.cos(pi), math.sin(pi))])
        elif k_value == 10:
            psi_vector = np.array([math.cos(3*pi/10), math.sin(3*pi/10) * complex(math.cos(pi), math.sin(pi))])
        elif k_value == 11:
            psi_vector = np.array([math.cos(4*pi/10), math.sin(4*pi/10) * complex(math.cos(pi), math.sin(pi))])
        elif k_value == 12:
            psi_vector = np.array([math.cos(pi/10), math.sin(pi/10) * complex(math.cos(3*pi/2), math.sin(3*pi/2))])
        elif k_value == 13:
            psi_vector = np.array([math.cos(2*pi/10), math.sin(2*pi/10) * complex(math.cos(3*pi/2), math.sin(3*pi/2))])
        elif k_value == 14:
            psi_vector = np.array([math.cos(3*pi/10), math.sin(3*pi/10) * complex(math.cos(3*pi/2), math.sin(3*pi/2))])
        else:  # k_value == 15
            psi_vector = np.array([math.cos(4*pi/10), math.sin(4*pi/10) * complex(math.cos(3*pi/2), math.sin(3*pi/2))])

    else:
        raise ValueError("When using 3d encoding num_bits must be at most 4.")

    return psi_vector


def get_brun_fig2_encoding(k_value, num_bits):
    if num_bits != 2:
        raise ValueError("When using brun_fig2 encoding num_bits must be set to 2.")

    if k_value == 0:
        psi = [1, 0]
    elif k_value == 1:
        psi = [0, 1]
    elif k_value == 2:
        psi = [
            math.cos(pi / 4),
            math.sin(pi / 4)
        ]
    else:  # k_value == 3:
        psi = [
            math.cos(3 * pi / 4),
            math.sin(3 * pi / 4)
        ]
    return np.array(psi)


def get_psi_with_ancillas(psi_vector, num_qubits):
    """
    Append to the statevector psi a number num_qubits - 1 of ancillary qubit states all set to |0>.

    :param psi_vector: The numpy array containing psi state in statevector representation.
    :param num_qubits: The total number of qubits the final state vector is composed of.
    :return: A numpy array containing the statevector representation of the state |psi>|0>|0>...|0>
    """
    ancillas_state = np.zeros(2**(num_qubits - 1), dtype=complex)
    ancillas_state[0] = complex(real=1.)
    psi_with_ancillas = np.tensordot(psi_vector, ancillas_state, axes=0).reshape(1, 2**num_qubits)

    # print(psi_with_ancillas) # DEBUG
    return psi_with_ancillas

