import numpy as np
from enum import Enum

import seaborn as sns
import matplotlib.pyplot as plt


class Failure_Type(Enum):
    DECODER = 1,
    LOGIC_X =  2,
    LOGIC_Z = 3 

def generate_code_input_state(code_size: int, x_error_rate: float, z_error_rate: float):
    """generate a surface code in a state resulting from syndrome measurement.

    Args:
        code_size (int): size of the code
        x_error_rate (float): x error rate [0, 1]
        z_error_rate (float): z error rate [0, 1]

    Return:
        syndrome: grid with the syndrome, typically what a decoder input my look like
        x_error_position : position of x errors
        z_error_position : position of z errors
        logic_x : set of qubits position for x logical operator
        logic_z : set of qubits position for z logical operator
    """
    #Grid that represent all the qubits in the surface code 
    grid = np.zeros((2*code_size-1, 2*code_size-1), dtype = np.uint8)

    # The grid represent the following surface code (X, Z : measure corresp to stabilizer, D: data qubits)
    #   D   Z   D   Z   D   Z   ... D  
    #   X   D   X   D   X   D   ... X
    #   D   Z   D   Z   D   Z   ... D
    #   X   D   X   D   X   D   ... X
    #   D   Z   D   Z   D   Z   ... D
    #   ...
    #   X   D   X   D   X   D   ... X
    #   D   Z   D   Z   D   Z   ... D  


    rows, cols = grid.shape
    i, j = np.indices((rows, cols))

    # Define the logical operator as set of data qubits

    lx = np.where(((i+j)%2 == 0) & (i == code_size - (code_size%2))) # first clause is to select data qubits, second one defines a line on an index where it's Z stab, and chosen arbirtraly as clause to the middle of the surface.
    logic_x = set(zip(lx[0], lx[1]))

    lz = np.where(((i+j)%2 == 0) & (j==code_size - (code_size%2))) # same idea, select a column.
    logic_z = set(zip(lz[0], lz[1]))


    #create grid filled with 0 with random position fixed to 2 for X error (according to rate), also set dtype to uint8 to optimize memory usage
    choices = np.array([0, 2], dtype=np.uint8)
    grid_x = np.random.choice(choices, size=(2*code_size - 1, 2*code_size - 1), p=[1 - x_error_rate, x_error_rate])

    #same for z error but set value to 3 at error position
    choices = np.array([0, 3], dtype=np.uint8)
    grid_z = np.random.choice(choices, size=(2*code_size - 1, 2*code_size - 1), p=[1 - z_error_rate, z_error_rate])

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

def check_correction(code_size, x_corrections, z_corrections, x_errors, z_errors, logical_x, logical_z) -> bool:
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

    g=np.zeros((2*code_size-1, 2*code_size-1), dtype = np.uint8)

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

    move_targets = []
    for idx, (x, y) in enumerate(anyons_pos):

        if random_decisions[idx] <= 0.5:
            move_targets.append((x,y))
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
                move_targets.append(new_position)
            else:
                move_targets.append((x,y))

    for (tx, ty) in move_targets:
        np.add.at(n_anyons, (tx, ty), 1)

    return n_anyons % 2, moves

def evolve_system(field, anyons_grid, tao_max, boundaries, axis, c_step =0.2):
    moves = []
    c = 1
    for t in range(tao_max):
        c = c + c_step
        for _ in range(int(c)):
            field = evolve_field(field, anyons_grid)
        anyons_grid, n_m = evolve_anyons(anyons_grid, field)

        moves.extend(n_m)

        # Break if no anyons left in the surface of interest
        if axis%2 == 1:
            if(np.all(anyons_grid[:, boundaries[0]:boundaries[1]] == 0)):
                break
        else:
            if(np.all(anyons_grid[boundaries[0]:boundaries[1], :] == 0)):
                break

    return moves, t


def get_correction_from_moves(moves, boundaries, transpose=False):
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


def gen_plot(df_results):
    plt.figure(figsize=(8, 6))
    sns.lineplot(data=df_results, x='error_rate', y='failure_rate', hue='code_size', marker="o")
    plt.title("Decoder Fail Rate vs Data Qubit Error Rate")
    plt.xlabel("Qubits Error Rate")
    plt.ylabel("Failure Rate")
    plt.grid(True)
    plt.legend(title="Code Size")
    plt.tight_layout()
    plt.savefig("final_outputs/fail_rate_vs_error_rate_by_code_size2.png")
    plt.close()

    failure_avg = df_results.groupby('error_rate')[['dfc_rate', 'lfcx_rate', 'lfcz_rate']].mean().reset_index()

    plt.figure(figsize=(8, 6))
    sns.lineplot(data=failure_avg, x='error_rate', y='dfc_rate', label='DFC Rate', marker="o")
    sns.lineplot(data=failure_avg, x='error_rate', y='lfcx_rate', label='LFCX Rate', marker="o")
    sns.lineplot(data=failure_avg, x='error_rate', y='lfcz_rate', label='LFCZ Rate', marker="o")
    plt.title("Average Failure Rates vs Error Rate")
    plt.xlabel("Error Rate")
    plt.ylabel("Failure Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("final_outputs/avg_failure_types_vs_error_rate2.png")
    plt.close()

    plt.figure(figsize=(8, 6))

    # Add total convergence time column if not already done
    df_results['total_convergence_time'] = df_results['avg_convergence_time_x'] + df_results['avg_convergence_time_z']

    sns.lineplot(
        data=df_results,
        x='error_rate',
        y='total_convergence_time',
        hue='code_size',
        marker="o"
    )

    plt.title("Total Convergence Time vs Error Rate")
    plt.xlabel("Error Rate")
    plt.ylabel("Convergence Time (X + Z)")
    plt.grid(True)
    plt.legend(title="Code Size")
    plt.tight_layout()
    plt.savefig("final_outputs/total_convergence_time_vs_error_rate_by_code_size2.png")
    plt.close()



