from numba import *
import numpy as np
from decoder_util import *
from numba.typed import List
import math

@njit(parallel=True)
def evolve_field_numba(grid, anyons_grid, smoothing=0.5):
    rows, cols = grid.shape
    padded = np.zeros((rows + 2, cols + 2))
    padded[1:-1, 1:-1] = grid  # Manual zero padding

    new_grid = np.empty_like(grid)

    for i in prange(rows):  # Use prange for outer loop (parallelized)
        for j in range(cols):
            top    = padded[i, j + 1]
            bottom = padded[i + 2, j + 1]
            left   = padded[i + 1, j]
            right  = padded[i + 1, j + 2]

            neighbors_avg = (1 - smoothing) * (top + bottom + left + right) / 4.0
            new_grid[i, j] = smoothing * grid[i, j] + neighbors_avg + anyons_grid[i, j]

    return new_grid

@njit
def evolve_anyons_numba(anyons, field):
    rows, cols = field.shape
    n_anyons = np.zeros_like(anyons)

    max_anyons = rows * cols
    moves = np.empty((max_anyons, 4), dtype=np.int32)
    move_count = 0

    for x in range(rows):
        for y in range(cols):
            if anyons[x, y] != 1:
                continue

            r = np.random.random()

            # Stay in place
            if r <= 0.5:
                tx, ty = x, y
            else:
                best_val = -1.0
                tx, ty = x, y  # fallback

                # Top
                if x > 0 and field[x-1, y] > best_val:
                    best_val = field[x-1, y]
                    tx, ty = x-1, y
                # Bottom
                if x < rows - 1 and field[x+1, y] > best_val:
                    best_val = field[x+1, y]
                    tx, ty = x+1, y
                # Left
                if y > 0 and field[x, y-1] > best_val:
                    best_val = field[x, y-1]
                    tx, ty = x, y-1
                # Right
                if y < cols - 1 and field[x, y+1] > best_val:
                    best_val = field[x, y+1]
                    tx, ty = x, y+1

                # Save move (from x,y to tx,ty)
                moves[move_count] = np.array([x, y, tx, ty])
                move_count += 1

            # Move anyon to target
            n_anyons[tx, ty] += 1

    m_out = [list(zip(x[::2], x[1::2])) for x in moves[:move_count]]
    return n_anyons % 2, m_out

def evolve_system_numba1(field, anyons_grid, tao_max, boundaries, axis, c_step =0.2):
    moves = []
    c = 1
    is_clean = False
    for t in range(tao_max):
        c = c + c_step
        for _ in range(int(c)):
            field = evolve_field_numba(field, anyons_grid)
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

def evolve_system_numba2(field, anyons_grid, tao_max, boundaries, axis, c_step =0.2):
    moves = []
    c = 1
    is_clean = False
    for t in range(tao_max):
        c = c + c_step
        for _ in range(int(c)):
            field = evolve_field_numba(field, anyons_grid)
        anyons_grid, n_m = evolve_anyons_numba(anyons_grid, field)

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

def numba_decode(syndrome_grid, mirror_factor = 3, retry = 4, tao_max = None, c_step = 0.2):
    if tao_max is None:
        tao_max = 5 * math.floor(math.log2(syndrome_grid.shape[0]/2)**2.5)
    
    # Extract anyons grid corresponding to each error
    xcopy = syndrome_grid.copy()
    zcopy = syndrome_grid.copy()
    z_anyons = zcopy[::2, 1::2]
    x_anyons = xcopy[1::2, ::2]

    

    # Extend z_anyons horizontally
    z_anyons, lr_bound = extend_with_symmetry(z_anyons, mirror_factor, 1)

    # Extend x_anyons vertically
    x_anyons, tb_bound = extend_with_symmetry(x_anyons, mirror_factor, 0)
    
    # Init "heat"-fields
    x_field = np.zeros(x_anyons.shape)
    z_field = np.zeros(z_anyons.shape)

    # Evolve fields and anyons until convergence
    is_x_clean = False
    retry_x = retry
    while(not is_x_clean and retry_x>0):
        # Init "heat"-fields
        x_field = np.zeros(x_anyons.shape)
        x_moves, x_convergence_time, is_x_clean = evolve_system_numba1(x_field, x_anyons.copy(), tao_max, tb_bound, 0)
        retry_x = retry_x - 1
    
    is_z_clean = False
    retry_z = retry
    while(not is_z_clean and retry_z>0):
        z_field = np.zeros(z_anyons.shape)
        z_moves, z_convergence_time, is_z_clean = evolve_system_numba1(z_field, z_anyons.copy(), tao_max, lr_bound, 1)
        retry_z = retry_z - 1


    # Compute the correction to apply according to the moves
    # Reminder, x_anyons are generated by z errors (and vice-versa) due to anti commutation prop. of stabilizer. Therefore, moves of x_anyons implies z_correction
    x_correction = get_correction_from_moves(z_moves, lr_bound)
    z_correction = get_correction_from_moves(x_moves, tb_bound, transpose=True)

    return x_correction, z_correction, x_convergence_time, z_convergence_time, retry

def numba_decode_2(syndrome_grid, mirror_factor = 3, retry = 4, tao_max = None, c_step = 0.2):
    if tao_max is None:
        tao_max = 5 * math.floor(math.log2(syndrome_grid.shape[0]/2)**2.5)
    
    # Extract anyons grid corresponding to each error
    xcopy = syndrome_grid.copy()
    zcopy = syndrome_grid.copy()
    z_anyons = zcopy[::2, 1::2]
    x_anyons = xcopy[1::2, ::2]

    

    # Extend z_anyons horizontally
    z_anyons, lr_bound = extend_with_symmetry(z_anyons, mirror_factor, 1)

    # Extend x_anyons vertically
    x_anyons, tb_bound = extend_with_symmetry(x_anyons, mirror_factor, 0)
    
    # Init "heat"-fields
    x_field = np.zeros(x_anyons.shape)
    z_field = np.zeros(z_anyons.shape)

    # Evolve fields and anyons until convergence
    is_x_clean = False
    retry_x = retry
    while(not is_x_clean and retry_x>0):
        # Init "heat"-fields
        x_field = np.zeros(x_anyons.shape)
        x_moves, x_convergence_time, is_x_clean = evolve_system_numba2(x_field, x_anyons.copy(), tao_max, tb_bound, 0)
        retry_x = retry_x - 1
    
    is_z_clean = False
    retry_z = retry
    while(not is_z_clean and retry_z>0):
        z_field = np.zeros(z_anyons.shape)
        z_moves, z_convergence_time, is_z_clean = evolve_system_numba2(z_field, z_anyons.copy(), tao_max, lr_bound, 1)
        retry_z = retry_z - 1


    # Compute the correction to apply according to the moves
    # Reminder, x_anyons are generated by z errors (and vice-versa) due to anti commutation prop. of stabilizer. Therefore, moves of x_anyons implies z_correction
    x_correction = get_correction_from_moves(z_moves, lr_bound)
    z_correction = get_correction_from_moves(x_moves, tb_bound, transpose=True)

    return x_correction, z_correction, x_convergence_time, z_convergence_time, 1


