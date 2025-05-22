import numpy as np
from decoder_util import *
import pymatching
from scipy.sparse import lil_matrix, csc

# ============================================
# File: decode_pymatch_util.py
# Author: Lenny Del Zio
# Date: 2025-05-18
# Description: Some utilities function to make it easy to run pymatching from our encoding of the data
# ============================================

def get_x_check_matrix(d):
    """Create parity check matrix for X stabilizer measurement.

    Args:
        d (int): code distance

    Returns:
        csc matrix
    """
    #   D1   Z   D2   Z   D3
    #   X1   D4   X   D5    X
    #   D6   Z   D7   Z D8
    #   X   D9   X   D10    X
    #   D11 Z   D12 Z   D13
    H_x = lil_matrix((d*(d-1), d**2+(d-1)**2), dtype=np.uint8)
    
    for i in range(d-1):
        for j in range(d):
            x_n = i*d + j
            t = j + i*(d+d-1)
            H_x[x_n, t] = 1 #qubit above
            b = j + (i+1)*(d+d-1)
            H_x[x_n, b] = 1 #qubit below
            if j!= 0:
                l = d+j-1 + i*(d+d-1) #qubit on the left
                H_x[x_n, l] = 1
            if j!= d-1:
                r = d+j + i*(d+d-1) #qubit on the righg
                H_x[x_n, r] = 1
    
    return H_x.tocsc()

def get_z_check_matrix(d):
    """Create parity check matrix for Z stabilizer measurement.

    Args:
        d (int): code distance

    Returns:
        csc matrix
    """
    H_z = lil_matrix((d*(d-1), d**2+(d-1)**2), dtype=np.uint8)
    for i in range(d):
        for j in range(d-1):
            z_n = i*(d-1)+j
            if i != 0:
                #top
                t = j + d +  (d+d-1) * (i-1)
                H_z[z_n, t] = 1
            if i != d-1:
                #bot
                b = j + d + (d+d-1) * i
                H_z[z_n, b] = 1
            l= j + (d+d-1) * i
            H_z[z_n, l] = 1
            r = j+1 + (d+d-1) * i
            H_z[z_n, r] = 1
    return H_z.tocsc()


def translate_correction_into_positions(syndrom_list, d):
    """Translate the outcome of pymatching into our grid representation

    Args:
        x_list (list): pymatching's prediction of the errors position
        d (int): Code distance

    Returns:
       corr: set of position to apply correction to.
    """
    filled = [elem for pair in zip(syndrom_list, [2]*len(syndrom_list)) for elem in pair][:-1] if len(syndrom_list) else []
    reshaped = np.array(filled).reshape((2*d -1, 2*d -1))
    corr = set(map(tuple, np.argwhere(reshaped == 1)))
    return corr

def init_pymatch(d):
    # Build check matrix for a given code distance
    H_x = get_x_check_matrix(d)
    matcher_x = pymatching.Matching(H_x)
    
    H_z = get_z_check_matrix(d)
    matcher_z = pymatching.Matching(H_z)
    
    return matcher_x, matcher_z


