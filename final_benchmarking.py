from concurrent.futures import ProcessPoolExecutor, as_completed
from final_decoder_util import *
from final_decoder import *
import time
import pandas as pd

def test_1_instance(args):
    success_count = 0
    fail_decoder = 0
    fail_logic_x = 0
    fail_logic_z = 0
    tdecod = 0.0
    tcheck = 0.0

    code_size, x_error_rate, z_error_rate = args

    syndrome_grid, x_error_positions, z_error_positions, logic_x, logic_z = generate_code_input_state(code_size, x_error_rate, z_error_rate)
    t1 = time.time()
    corr_x, corr_z, tx, tz = decode(syndrome_grid)
    t2 = time.time()
    tdecod += t2-t1

    t3 = time.time()
    is_succeed, failures = check_correction(code_size, corr_x, corr_z, x_error_positions, z_error_positions, logic_x, logic_z)
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

    return success_count, fail_decoder, fail_logic_x, fail_logic_z, tx, tz, tdecod, tcheck

def main():
    # Benchmark parameters
    code_size = [10, 30, 50, 100]               # Lattice sizes
    error_rate = np.linspace(0.07, 0.09, 10)  # Error rates
    samples_per_point = 1000       # Simulations per (l, e)

    results = []
    

    print("Running simulations")

    with ProcessPoolExecutor(max_workers = 20) as executor:
        for cs in code_size:
            for err in error_rate:
                args_list = [(cs, err, err)] * samples_per_point
                futures = executor.map(test_1_instance, args_list)


                total_dfc = 0
                total_lfcx = 0
                total_lfcz = 0
                total_dectime = 0
                total_checktime = 0
                convergence_time_x = []
                convergence_time_z = []
                success = 0

                for s, dfc, lfcx, lfcz, ctimex, ctimez, tdecod, tcheck in futures:
                    convergence_time_x.append(ctimex)
                    convergence_time_z.append(ctimez)
                    total_dfc += dfc
                    total_lfcx += lfcx
                    total_lfcz += lfcz
                    total_dectime += tdecod
                    total_checktime += tcheck
                    if s:
                        success += 1
                
                results.append({
                    'code_size': cs,
                    'error_rate': err,
                    'failure_rate': 1- success / samples_per_point,
                    'dfc_rate': total_dfc / samples_per_point,
                    'lfcx_rate': total_lfcx / samples_per_point,
                    'lfcz_rate': total_lfcz / samples_per_point,
                    'avg_decoding_time': total_dectime / samples_per_point,
                    'avg_check_time': total_checktime / samples_per_point,
                    'avg_convergence_time_x': np.mean(convergence_time_x),
                    'avg_convergence_time_z': np.mean(convergence_time_z)
                })
                print(f"{(cs, err)}, {samples_per_point} samples done in {total_checktime+total_dectime:.2f} seconds")

    df_results = pd.DataFrame(results)

    gen_plot(df_results)

if __name__ == '__main__':
    #check runtimes :
    # for i in [10, 30, 50, 100, 200]:
    #     success_count, fail_decoder, fail_logic_x, fail_logic_z, tx, tz, tdecod, tcheck = test_1_instance((i, 0.05, 0.05))
    #     print(f"code size : {i}, exec time : {tdecod+tcheck}")
    main()

