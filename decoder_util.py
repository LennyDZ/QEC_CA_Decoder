import json
import numpy as np
from enum import Enum

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix
from decode_pymatch_util import *
import pymatching
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================
# File: decoder_util.py
# Author: Lenny Del Zio
# Date: 2025-05-18
# Description: Many utilities functions for Cellular automata decoder for surface codes
# Some functions may lack explicit description. Usually the function's name is quite explicit.
# ============================================

class Failure_Type(Enum):
    """Describe the different types of failure that a CA decoder can encounter.
    """
    DECODER = 1
    LOGIC_X = 2
    LOGIC_Z = 3 
    
def generate_code_input_state(code_distance: int, x_error_rate: float, z_error_rate: float):
    """
        Generate a sample syndrome for a surface code of given distance, under bit-flip / phase-flip error.
        Each flips is applied independently to each data qubits with given rate.
        No measurement errors and no error on ancilla qubits are considered.
        
        The following layout is considered (X, Z : measurement qubits corresp to stabilizer, D: data qubits):
          D   Z   D   Z   D   Z   ... D  
          X   D   X   D   X   D   ... X
          D   Z   D   Z   D   Z   ... D
          X   D   X   D   X   D   ... X
          D   Z   D   Z   D   Z   ... D
          ...
          X   D   X   D   X   D   ... X
          D   Z   D   Z   D   Z   ... D  

    Args:
        code_distance (int): distance of the surface code
        x_error_rate (float): x error rate [0, 1]
        z_error_rate (float): z error rate [0, 1]

    Return:
        syndrome: grid with the syndrome, typically what a decoder input my look like, the grid corresponds to the whole layout. Index corresponding to data qubits will always be zero and index corresp. to stab. measurement have value corresp. to associated the parity check (i. e. 1 if it "detects" an error)
        x_error_position : position of x errors (for later check of correction validity)
        z_error_position : position of z errors (for later check of correction validity)
        logic_x : set of qubits position for x logical operator (arbitrary defined to be a straightforward chain)
        logic_z : set of qubits position for z logical operator (arbitrary defined to be a straightforward chain)
    """
    
    #Grid that represent all the qubits in the surface code 
    grid = np.zeros((2*code_distance-1, 2*code_distance-1), dtype = np.uint8)


    rows, cols = grid.shape
    i, j = np.indices((rows, cols))

    # Define the logical operator as set of data qubits

    lx = np.where(((i+j)%2 == 0) & (i == code_distance - (code_distance%2))) # first clause is to select data qubits, second one defines a line on an index where it's Z stab, and chosen arbirtraly as clause to the middle of the surface.
    logic_x = set(zip(lx[0], lx[1]))

    lz = np.where(((i+j)%2 == 0) & (j==code_distance - (code_distance%2))) # same idea, select a column.
    logic_z = set(zip(lz[0], lz[1]))


    #create grid filled with 0 with random position fixed to 2 for X error (according to rate), also set dtype to uint8 to optimize memory usage
    choices = np.array([0, 2], dtype=np.uint8)
    grid_x = np.random.choice(choices, size=(2*code_distance - 1, 2*code_distance - 1), p=[1 - x_error_rate, x_error_rate])

    #same for z error but set value to 3 at error position
    choices = np.array([0, 3], dtype=np.uint8)
    grid_z = np.random.choice(choices, size=(2*code_distance - 1, 2*code_distance - 1), p=[1 - z_error_rate, z_error_rate])

    grid += grid_x
    grid += grid_z

    #we error only on data qubits, so we set all measurement qubits to 0
    meas_qubits_mask = (i+j)%2 == 1
    grid[meas_qubits_mask] = 0 

    # Clumsy part, where we encode the stabilizer reaction to the error.
    for (x, y), v in np.ndenumerate(grid):
        #X error => Z stabilizer reacts
        if v==2 or v==5:
            # even line => Z stab are right and left of data qubit
            if x%2 == 0:
                if y!= 0:
                    grid[x][y-1] = (grid[x][y-1] + 1)%2
                if y!= cols-1:
                    grid[x][y+1] = (grid[x][y+1] + 1)%2
            else:
                if x!= 0:
                    grid[x-1][y] = (grid[x-1][y] + 1) %2
                if x!= rows-1:
                    grid[x+1][y] = (grid[x+1][y] + 1) %2
        #Z error => X stabiliter reacts
        if v==3 or v==5:
            if x%2==0:
                if x!= 0:
                    grid[x-1][y] = (grid[x-1][y] + 1) %2
                if x!= rows-1:
                    grid[x+1][y] = (grid[x+1][y] + 1) %2
            else:
                if y!= 0:
                    grid[x][y-1] = (grid[x][y-1] + 1)%2
                if y!= cols-1:
                    grid[x][y+1] = (grid[x][y+1] + 1)%2
    
    #List error positions
    x_error_positions = set(zip(*np.where((grid == 2) | (grid == 5))))
    z_error_positions = set(zip(*np.where((grid == 3) | (grid == 5))))

    syndrome_grid = grid.copy()
    data_qubits_mask = (i+j)%2 == 0
    
    syndrome_grid[data_qubits_mask] = 0

    return syndrome_grid, x_error_positions, z_error_positions, logic_x, logic_z

def check_correction(code_distance, x_corrections, z_corrections, x_errors, z_errors, logical_x, logical_z) -> bool:
    """Check wether a correction is valid or not.

    Args:
        x_corrections (set): set of data qubits position to apply x correction
        z_corrections (set): set of data qubits position to apply z correction
        x_errors (set): set of position of x errors
        z_errors (set): set of position of z errors
        logical_x (set): logical X operator of the code, as set of data qubits
        logical_z (set): logical Z operator of the code, as set of data qubits

    Returns:
        bool: wether the correction are succeed or not
    """
    failures = set()

    pauli_x_after_correction = x_corrections ^ x_errors
    pauli_z_after_correction = z_corrections ^ z_errors

    g=np.zeros((2*code_distance-1, 2*code_distance-1), dtype = np.uint8)

    for nx in pauli_x_after_correction:
        g[nx] = 2
    for nz in pauli_z_after_correction:
        if g[nz] == 0:
            g[nz] = 3
        elif g[nz] == 2:
            g[nz] = 5
    
    rows, cols = g.shape
    i, j = np.indices((rows, cols))
    for (x, y), v in np.ndenumerate(g):
        if v==2 or v==5:
            if x%2 == 0:
                if y!= 0:
                    g[x][y-1] = (g[x][y-1] + 1)%2
                if y!= cols-1:
                    g[x][y+1] = (g[x][y+1] + 1)%2
            else:
                if x!= 0:
                    g[x-1][y] = (g[x-1][y] + 1) %2
                if x!= rows-1:
                    g[x+1][y] = (g[x+1][y] + 1) %2
        if v==3 or v==5:
            if x%2==0:
                if x!= 0:
                    g[x-1][y] = (g[x-1][y] + 1) %2
                if x!= rows-1:
                    g[x+1][y] = (g[x+1][y] + 1) %2
            else:
                if y!= 0:
                    g[x][y-1] = (g[x][y-1] + 1)%2
                if y!= cols-1:
                    g[x][y+1] = (g[x][y+1] + 1)%2

    resulting_syndrome_grid = g.copy()
    data_qubits_mask = (i+j)%2 == 0
    resulting_syndrome_grid[data_qubits_mask] = 0

    if np.any(resulting_syndrome_grid != 0):
        failures.add(Failure_Type.DECODER)
    
    log_x_cross = (len(pauli_z_after_correction & logical_x))
    if log_x_cross%2 == 1:
        failures.add(Failure_Type.LOGIC_Z)

    log_z_cross = (len(pauli_x_after_correction & logical_z))
    if log_z_cross%2 == 1:
       failures.add(Failure_Type.LOGIC_X)
    
    if not failures:
        return True, failures
    else:
        return False, failures

def extend_with_symmetry(grid, expansion_factor, axis):
    # Extend with symmetry top-bottom for x_anyons and left-right for z_anyons
    grid_size = grid.shape[axis]
    int_m_size = int(expansion_factor * grid_size)
    full_replica_count = int_m_size // grid_size
    last_m_size = int_m_size % grid_size

    if axis%2==1:
        m_left =  np.empty((grid.shape[0], 0))
        m_right=  np.empty((grid.shape[0], 0))
        if last_m_size == 0:
            half_left = np.empty((grid.shape[0], 0))
            half_right = np.empty((grid.shape[0], 0))
        else:
            half_left = grid[:, :last_m_size]
            half_right = grid[:, last_m_size:]
    else:
        m_left =  np.empty((0, grid.shape[1]))
        m_right=  np.empty((0, grid.shape[1]))
        if last_m_size == 0:
            half_left = np.empty((0, grid.shape[1]))
            half_right = np.empty((0, grid.shape[1]))
        else:
            half_left = grid[:last_m_size, :]
            half_right = grid[last_m_size:, :]
    
    for i in range(full_replica_count):
        if axis%2 == 1:
            if i%2 == 0:
                m_left = np.hstack((np.fliplr(grid), m_left))
                m_right = np.hstack((m_right, np.fliplr(grid)))
            else:
                m_left = np.hstack((grid, m_left))
                m_right = np.hstack((m_right, grid))
        else:
            if i%2 == 0:
                m_left = np.vstack((np.flipud(grid), m_left))
                m_right = np.vstack((m_right, np.flipud(grid)))
            else:
                m_left = np.vstack((grid, m_left))
                m_right = np.vstack((m_right, grid))

    if full_replica_count%2 == 0:
        if axis%2==1:
            m_left = np.hstack((np.fliplr(half_left), m_left))
            m_right = np.hstack((m_right, np.fliplr(half_right)))
        else:
            m_left = np.vstack((np.flipud(half_left), m_left))
            m_right = np.vstack((m_right, np.flipud(half_right)))

    if axis%2 == 1:
        f_grid = np.hstack((m_left, grid, m_right))
        boundaries = [m_left.shape[1], m_left.shape[1] + grid.shape[1]]
    else:
        f_grid = np.vstack((m_left, grid, m_right))
        boundaries = [m_left.shape[0], m_left.shape[0] + grid.shape[0]]
    
    return f_grid, boundaries

def evolve_field(grid, anyons_grid, smoothing = 0.5):
    padded = np.pad(grid, pad_width=1, mode='constant', constant_values=0)

    top    = padded[:-2, 1:-1]
    bottom = padded[2:, 1:-1]
    left   = padded[1:-1, :-2]
    right  = padded[1:-1, 2:]

    neighbors_avg = (1 - smoothing) * (top + bottom + left + right) / 4

    new_grid = smoothing * grid + neighbors_avg + anyons_grid

    return new_grid

def evolve_anyons(anyons, field):
    n_anyons = np.zeros(anyons.shape)
    rows, cols = field.shape
    moves = []
    anyons_pos = np.argwhere(anyons == 1)

    random_decisions = np.random.random(len(anyons_pos))

    move_targets = np.empty((len(anyons_pos), 2), dtype=int)
    for idx, (x, y) in enumerate(anyons_pos):

        if random_decisions[idx] <= 0.5:
            move_targets[idx] = (x,y)
        else:
            neighbors_position = []

            # Top
            if x > 0:
                neighbors_position.append((x - 1, y))
            # Bottom
            if x < rows - 1:
                neighbors_position.append((x + 1, y))
            # Left
            if y > 0:
                neighbors_position.append((x, y - 1))
            # Right
            if y < cols - 1:
                neighbors_position.append((x, y + 1))

            if neighbors_position:
                neighbors = [field[n] for n in neighbors_position]
                max_neighbors_index = neighbors.index(max(neighbors))
                new_position = neighbors_position[max_neighbors_index]

                moves.append(((x, y), new_position))
                move_targets[idx] = new_position
            else:
                move_targets[idx] = (x,y)

    for (tx, ty) in move_targets:
        np.add.at(n_anyons, (tx, ty), 1)

    return n_anyons % 2, moves

def evolve_system(field, anyons_grid, tao_max, boundaries, axis, c_step =0.2):
    """Evolve the whole system until relevant grid area is empty of anyons

    Args:
        field (np.array): grid for the field evolution
        anyons_grid (np.array): grid for the anyons movements
        tao_max (int): max number of anyons steps
        boundaries ((int, int)): index limits of the relevant area
        axis (int): 0 or 1 depending if we perform the problem for X or Z syndrome
        c_step (float, optional): Step to increase field velocity between each anyons step. Defaults to 0.2.

    Returns:
        moves, t, is_clean: set of moves, step till complete, wether there is anyons left in the relevant area or not
    """
    moves = []
    c = 1
    is_clean = False
    for t in range(tao_max):
        c = c + c_step
        for _ in range(int(c)):
            field = evolve_field(field, anyons_grid)
        anyons_grid, n_m = evolve_anyons(anyons_grid, field)

        moves.extend(n_m)

        # Break if no anyons left in the surface of interest
        if axis%2 == 1:
            if(np.all(anyons_grid[:, boundaries[0]:boundaries[1]] == 0)):
                is_clean = True
                break
        else:
            if(np.all(anyons_grid[boundaries[0]:boundaries[1], :] == 0)):
                is_clean = True
                break

    return moves, t, is_clean

def get_correction_from_moves(moves, boundaries, transpose=False):
    """
        Get the corrections that have to be applied according to the anyons' moves.

    Args:
        moves (set((from, to)): anyons moves ("from" and "to" are pair of x,y coordinate (index in the grid) of the initial/final position of each moves (of 1 step always). 
        boundaries (left/top, right/bottom): in the extended dimension, this are index of the limits of the "initial" grid (that is the grid before it gets extended). So moves outsides these boundaries are of no interest for the correction
        transpose (bool, optional): Wether we have to transpose the analysis or not (the problem for X-syndrome is the same as the one for Z-syndrome up to matrix transpose). Defaults to False.

    Returns:
        _type_: _description_
    """
    correction = set()

    if transpose :
        moves = [((y1, x1), (y2, x2)) for ((x1, y1), (x2, y2)) in moves]

    size = boundaries[1]-boundaries[0]

    for (x1,y1) , (x2, y2) in moves:
            # Compute postion in the original grid (without extension)
            ny1 = y1-boundaries[0]
            ny2 = y2-boundaries[0]
            if 0 <= ny1 < size and 0 <= ny2 < size:
                #inside ruban, normal case
                if x1!=x2:
                    ey = 2*ny1 + 1
                    if x1!= 0 and x2!= 0:
                        ex = 2*x1+1 if x1<x2 else 2*x2+1
                    elif x1==1 or x2==1:
                        ex = 1
                    else:
                        ex=0
                if ny1!=ny2:
                    ey = 2*ny1 if ny1>ny2 else 2*ny2
                    ex = 2*x1
            else:
                if x1!=x2:
                    ex=-1
                    ey=-1
                else:
                    ex = 2*x1
                    if (ny1, ny2)==(-1, 0) or (ny1, ny2)==(0, -1):
                        ey = 0
                    elif (ny1, ny2)==(size, size-1) or (ny1, ny2)==(size-1, size):
                        ey= size*2
                    else:
                        ex=-1
                        ey=-1
            e=(ex, ey)
            if e != (-1,-1):
                if e in correction:
                    correction.remove(e)
                else:
                    correction.add(e)
            
    if transpose :
        correction = {(y, x) for (x,y) in correction}

    return correction

def run_pymatch_decoder():
    """
        Just run an instance of pymatching decoder
        This is leftover code from a previous phase or purpose — it no longer serves an active role.
    """
    d = 5
    n_qubits = d * d
    n_stabilizers = (d - 1) * d  # 4 * 5 = 20

    # Create a sparse matrix in LIL format (good for construction)
    H_z = lil_matrix((n_stabilizers, n_qubits), dtype=np.uint8)

    # Fill the matrix
    row = 0
    for r in range(d - 1):
        for c in range(d):
            q1 = r * d + c
            q2 = (r + 1) * d + c
            H_z[row, q1] = 1
            H_z[row, q2] = 1
            row += 1

    # Convert to CSC format for pymatching
    H_z = H_z.tocsc()

    # Now use PyMatching to create the decoder
    matching = pymatching.Matching(H_z)

    # Simulate a simple error: X error on qubit 6
    error = np.zeros(n_qubits, dtype=np.uint8)
    error[6] = 1

    # Compute syndrome
    syndrome = H_z @ error % 2

    # Decode
    correction = matching.decode(syndrome)

    print("Syndrome:   ", syndrome.astype(int))
    print("Correction: ", correction.astype(int))
    print("Residual error: ", (error + correction) % 2)

def test_1_instance(args):
    """Test one sample of a decoding process

    Args:
        args (Tuple): code_distance, x_error_rate, z_error_rate, mirror_size, max_tao, retry, decode_func

    Returns:
        Tuple(success_count, fail_decoder, fail_logic_x, fail_logic_z, tx, tz, tdecod, tcheck, retry_count): result of the test, all are int/float for convenience in later analysis.
    """
    success_count = 0
    fail_decoder = 0
    fail_logic_x = 0
    fail_logic_z = 0
    tdecod = 0.0
    tcheck = 0.0

    # unpack the args
    code_distance, x_error_rate, z_error_rate, mirror_size, max_tao, max_retry, decode_func = args

    # Generate input sample
    syndrome_grid, x_error_positions, z_error_positions, logic_x, logic_z = generate_code_input_state(code_distance, x_error_rate, z_error_rate)
    t1 = time.time()
    # Decode
    corr_x, corr_z, tx, tz, retry_count = decode_func(syndrome_grid, retry = max_retry, mirror_factor=mirror_size, tao_max = max_tao)
    t2 = time.time()
    tdecod += t2-t1

    t3 = time.time()
    # Check solution
    is_succeed, failures = check_correction(code_distance, corr_x, corr_z, x_error_positions, z_error_positions, logic_x, logic_z)
    t4 = time.time()
    tcheck += t4-t3
    
    if is_succeed:
        success_count += 1
    else:
        if Failure_Type.DECODER in failures:
            fail_decoder += 1
        if Failure_Type.LOGIC_X in failures:
            fail_logic_x += 1
        if Failure_Type.LOGIC_Z in failures:
            fail_logic_z += 1

    return success_count, fail_decoder, fail_logic_x, fail_logic_z, tx, tz, tdecod, tcheck, retry_count

def run_many_shot(cs, errx, erry, msize, maxtao, samples_per_point, decode_func, retry=4,):
    results = []
    with ProcessPoolExecutor(max_workers = 20) as executor:
        args_list = [(cs, errx, erry, msize, maxtao, retry, decode_func)] * samples_per_point
        futures = executor.map(test_1_instance, args_list)

        total_dfc = 0
        total_lfcx = 0
        total_lfcz = 0
        total_dectime = 0
        total_checktime = 0
        convergence_time_x = []
        convergence_time_z = []
        success = 0
        retry_count = 0

        for s, dfc, lfcx, lfcz, ctimex, ctimez, tdecod, tcheck, retry in futures:
            convergence_time_x.append(ctimex)
            convergence_time_z.append(ctimez)
            total_dfc += dfc
            total_lfcx += lfcx
            total_lfcz += lfcz
            total_dectime += tdecod
            total_checktime += tcheck
            retry_count += retry
            success += s

        
        results = {
            'code_distance': cs,
            'mirror_size': msize,
            'maxtao': maxtao,
            'error_rate': errx,
            'failure_rate': 1 - success / samples_per_point,
            'dfc_rate': total_dfc / samples_per_point,
            'lfcx_rate': total_lfcx / samples_per_point,
            'lfcz_rate': total_lfcz / samples_per_point,
            'avg_decoding_time': total_dectime / samples_per_point,
            'avg_check_time': total_checktime / samples_per_point,
            'avg_convergence_time_x': np.mean(convergence_time_x),
            'avg_convergence_time_z': np.mean(convergence_time_z),
            'avg_decoder_trial':retry_count/samples_per_point
        }
        print(f"{(cs, errx)}, {samples_per_point} samples done in {total_checktime+total_dectime:.2f} seconds")

        return results
    
def test_different_implementation_version(code_distance, error_rate, matcher_x, matcher_z, implementations):
    """For one code size, one error rate, test the same input sample with each decoder.

    Args:
        code_distance (int)
        error_rate (float): [0,1]
        matcher_x (pymatching.matcher)
        matcher_z (pymatching.matcher)

    Returns:
        Obejct: implementation(string), time(float, in seconds), sucess(boolean), code_distance(int)
    """
    syndrome_grid, x_error_positions, z_error_positions, logic_x, logic_z = generate_code_input_state(code_distance, error_rate, error_rate)
    
    results = []

    for name, decoder_func in implementations:
        t1 = time.time()
        corr_x, corr_z, tx, tz, retry_count = decoder_func(syndrome_grid)
        t2 = time.time()
        tdecod = t2 - t1
        check = check_correction(code_distance, corr_x, corr_z, x_error_positions, z_error_positions, logic_x, logic_z)
        # ✅ Store results in a structured way
        results.append({
            'implementation': name,
            'time': tdecod,
            'success': check[0],
            'code_distance': code_distance
        })
    
    # Pymatching :
    # Translate syndrome
    x_anyons = syndrome_grid.copy()[1::2, ::2]
    flatten_x_syndrome = x_anyons.flatten().tolist()
    x_syndrome = np.array(flatten_x_syndrome, dtype=np.uint8)
    z_anyons = syndrome_grid.copy()[::2, 1::2]
    flatten_z_syndrome = z_anyons.flatten().tolist()
    z_syndrome = np.array(flatten_z_syndrome, dtype=np.uint8)
    # Decode
    ts = time.time()
    predicted_errors_z = matcher_x.decode(x_syndrome)
    predicted_errors_x = matcher_z.decode(z_syndrome)
    te = time.time()

    # Translate pymatching result in set of positions
    corr_x = translate_correction_into_positions(predicted_errors_x, code_distance)
    corr_z = translate_correction_into_positions(predicted_errors_z, code_distance)
    
    check = check_correction(code_distance, corr_x, corr_z, x_error_positions, z_error_positions, logic_x, logic_z)
    # ✅ Store results in a structured way
    results.append({
        'implementation': 'PyMatching',
        'time': te-ts,
        'success': check[0],
        'code_distance': code_distance
    })
    return results
