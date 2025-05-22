from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import numpy as np
from decoder_util import *
from numba_decoder import *
from numpy_decoder import *
# from benchmark_plot import *
from decode_pymatch_util import *
import pymatching
import time
from collections import defaultdict
import pandas as pd
import os

# ============================================
# File: benchmarking.py
# Author: Lenny Del Zio
# Date: 2025-05-18
# Description: Set of function to evaluate different aspects of the decoder and generate plots
# ============================================

# List and ref all the different implementation of CA decoder
DECODER_IMPLEMENTATION = [("numpy vectoritze", decode), ("numba on field only", numba_decode), ("numba on field and anyons", numba_decode_2)]

def benchmark_replicas_sizes():
    """
        Evaluates the performance of the decoder for different factor of the size of the expansion of the grids.
        Resulting plots shows execution time and success rates vs code distance, for each implementation.
        Save plots and data in "test_outputs/replicas_test/"
    """
    # Benchmark parameters
    code_distance = 20                     # Code sizes
    error_range = [0.01, 0.09]
    error_rate_sample = 10
    error_rate = np.linspace(error_range[0], error_range[1] , error_rate_sample)  # Simulated Flip-Error rates
    samples_per_point = 10000            # Simulations shots per point
    mirror_size = [0, 0.5, 1, 1.5, 2, 3, 10]
    mtao_fac = 4
    max_tao = lambda cs : mtao_fac * math.floor(math.log2(cs)**2.5)
    results = []
    
    print(f"Running simulations...")
    for m in mirror_size:
        print(f" start m = {m}")
        for err in error_rate:
            if m==1:
                r = run_many_shot(code_distance, err, err, m, max_tao(code_distance), samples_per_point, decode, retry = 1)
            else :
                r = run_many_shot(code_distance, err, err, m, max_tao(code_distance), samples_per_point, decode)
            results.append(r)
    df_results = pd.DataFrame(results)

    print("Simulations completed, process data and plots...")
    
    try:
        run_spec = {'num_shots': samples_per_point, "max_tao":max_tao(code_distance), "mirror_size":mirror_size, 'error_range':error_rate.tolist(), 'num_error_rate_sample': error_rate_sample, 'code_distances': code_distance, 'retry':4}

        
        os.makedirs("test_outputs", exist_ok=True)
        os.makedirs("test_outputs/replicas_test", exist_ok=True)
        # Create a dir from the hash of the spec
        encoded = json.dumps(run_spec, sort_keys=True).encode()
        hashed_dir = hashlib.sha256(encoded).hexdigest()
        short_hash = hashed_dir[:8]
        output_dir = "test_outputs/replicas_test/" + short_hash + "mirror2" 
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/test_spec.txt", "w") as file:
            json.dump(run_spec, file, indent=4)
            file.write("\n")
            file.write("\n")
            file.write("Data Structure :\n")
            file.write("----------------\n")
            file.write(df_results.dtypes.to_string()) 
        
        df_results.to_pickle(f"{output_dir}/test_data")
        plot_mirror(df_results, output_dir=output_dir)
        print("Done !")
    except Exception as e:
        print(f"Something went wrong: {e}")
        df_results.to_pickle(f"rescue_saving/data")
        print(f"Saved execution data in rescue_saving/data, Warning! These data will be overwrite by another failed test.")    

def benchmark_implementations():
    """
        Test the differents implementations of cellular automata decoder, which are defined in DECODER_IMPLEMENTATION, also test with PyMatching MWPM implementations.
        Test are desinged to be executed for a fixed error rate, for multiple code distances.
        Resulting plots shows execution time and success rates vs code distance, for each implementation.
        Save plots and data in "test_outputs/error_reduction/"
        Other parameters include the number of retry allowed if decoder failure happen, size of the replicas added (ref. to doc), max number of anyons movement steps allowed
    """
    # Benchmark parameters
    code_distances = [5, 10, 20, 30, 50, 70, 80, 100]
    error_rate = 0.07  # Simulated Flip-Error rates
    ## mirror_size = 3 ## default
    # decoder_retry = 4 ## Default
    mtao_fac = 5  ## Default
    max_tao = lambda cs : mtao_fac * math.floor(math.log2(cs)**2.5) ## Default
    
    sample_per_implementation = 300
    
    print("Benchmark different implementations for different code distances...")
    
    all_results = []
    for cs in code_distances:
        print(f"     Test implementation for code_distance = {cs}")
        # Temporary storage to compute average per implementation
        

        mx, mz  = init_pymatch(cs)
        
        for _ in range(sample_per_implementation):
            results = test_different_implementation_version(cs, error_rate, mx, mz, DECODER_IMPLEMENTATION)
            for res in results:
                all_results.append(res)

    df = pd.DataFrame(all_results)

    df_ts = time.time()
    summary_df = df.groupby(['implementation', 'code_distance']).agg(
        avg_time=('time', 'mean'),
        success_rate=('success', lambda x: sum(x) / len(x)),
        total_shots=('success', 'count')
    ).reset_index()
    df_te = time.time()
    
    print("Execution successfully complete, process data and plot...")
    try:
        run_spec = {'num_shots': sample_per_implementation, "max_tao":[max_tao(cs) for cs in code_distances], 'error_rate':error_rate, 'code_distances': code_distances}
        
        os.makedirs("test_outputs", exist_ok=True)
        os.makedirs("test_outputs/implementation_test", exist_ok=True)
        # Create a dir from the hash of the spec
        encoded = json.dumps(run_spec, sort_keys=True).encode()
        hashed_dir = hashlib.sha256(encoded).hexdigest()
        short_hash = hashed_dir[:8]
        output_dir = "test_outputs/implementation_test/" + short_hash + "implem"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/test_spec.txt", "w") as file:
            json.dump(run_spec, file, indent=4)
            file.write("\n")
            file.write("\n")
            file.write("Data Structure :\n")
            file.write("----------------\n")
            file.write(summary_df.dtypes.to_string()) 
        
        summary_df.to_pickle(f"{output_dir}/data")
        plot_implementation_benchmark(summary_df, output_dir)
        print("Done !")
    except Exception as e:
        print(f"Something went wrong: {e}")
        summary_df.to_pickle(f"rescue_saving/data")
        print(f"Saved execution data in rescue_saving/data, Warning! These data will be overwrite by another failed test.")
    
def evaluate_error_reduction(p = 0.07):
    """
        This is design to assert the decoder's capacity to reduce error exponentially when increasing code distance, for a fixed error rate.
        Other parameters can be tuned as desired
        Plot the success rate vs code size, over a log scale, save plot in "test_outputs/error_reduction/"
        
        Args:
            p (float, optional): Error rate to 0.07.
    """
    code_distance = [5, 10, 20, 30, 50]
    error_rate = p
    samples_per_point = 1000                 # Simulations shots per point
    mirror_size = 3
    decoder_retry = 4
    mtao_fac = 5
    max_tao = lambda cs : mtao_fac * math.floor(math.log2(cs)**2.5)
    
    print(f"Running simulations, {samples_per_point} shots for each code size ...")
    results = []

    for cs in code_distance:
        print(f" Start simulation for distance-{cs} code")
        if cs<=30:
            r = run_many_shot(cs, error_rate, error_rate, mirror_size, max_tao(cs), samples_per_point, decode, retry =decoder_retry)
        else:
            r = run_many_shot(cs, error_rate, error_rate, mirror_size, max_tao(cs), samples_per_point, decode, retry =decoder_retry)
        results.append(r)
    df_results = pd.DataFrame(results)
    
    print("Simulations completed, process data and plots ...")
    try:
        run_spec = {'num_shots': samples_per_point, "max_tao":[max_tao(cs) for cs in code_distance], "mirror_size":mirror_size, 'error_rate':error_rate, 'code_distances': code_distance, 'retry': decoder_retry}
        
        os.makedirs("test_outputs", exist_ok=True)
        os.makedirs("test_outputs/error_reduction/", exist_ok=True)
        # Create a dir from the hash of the spec
        encoded = json.dumps(run_spec, sort_keys=True).encode()
        hashed_dir = hashlib.sha256(encoded).hexdigest()
        short_hash = hashed_dir[:8]
        output_dir = "test_outputs/error_reduction/" + short_hash + "exp_red"
        os.makedirs(output_dir, exist_ok=True)
        
        with open(f"{output_dir}/test_spec.txt", "w") as file:
            json.dump(run_spec, file, indent=4)
            file.write("\n")
            file.write("\n")
            file.write("Data Structure :\n")
            file.write("----------------\n")
            file.write(df_results.dtypes.to_string()) 
        
        plot_for_exp_red(df_results)
        df_results.to_pickle(f"{output_dir}/data")
        print("Done !")
    except Exception as e:
        print(f"Something went wrong: {e}")
        df_results.to_pickle(f"rescue_saving/data")
        print(f"Saved execution data in rescue_saving/data, Warning! These data will be overwrite by another failed test.")

def benchmark_success_rates():
    """
        Evaluate the decoder's success rates vs error rates, for different code distances.
        Resulting plots shows success rates vs code distance, frequency of each type of error and average number of anyons step to terminate.
        Save plots and data in "test_outputs/success_rate/"
    """
    # Benchmark parameters
    code_distance = [10, 20, 30, 50, 70, 100]                      # Code sizes
    error_range = [0.08, 0.09]
    error_rate_sample = 10
    error_rate = np.linspace(error_range[0], error_range[1] , error_rate_sample)  # Simulated Flip-Error rates
    error_rate = error_rate[(error_rate >= 0.07) & (error_rate <= 0.09)]
    samples_per_point = 1000                # Simulations shots per point
    mirror_size = 3
    decoder_retry = 4
    mtao_fac = 5
    max_tao = lambda cs : mtao_fac * math.floor(math.log2(cs)**2.5)
    
    sample_per_implementation = 10

    print("Running simulations ...")
    results = []

    for cs in code_distance:
        for err in error_rate:
            r = run_many_shot(cs, err, err, mirror_size, max_tao(cs), samples_per_point, decode, retry =decoder_retry)
            results.append(r)
    df_results = pd.DataFrame(results)
    
    print("Simulations completed, process data and plots ...")
    try:
        run_spec = {'num_shots': samples_per_point, "max_tao":[max_tao(cs) for cs in code_distance], "mirror_size":mirror_size, 'error_range':error_range, 'num_error_rate_sample': error_rate_sample, 'error_rates': error_rate, 'code_distances': code_distance, 'retry': decoder_retry}

        os.makedirs("test_outputs/success_rate", exist_ok=True)
        # Create a dir from the hash of the spec
        encoded = json.dumps(run_spec, sort_keys=True).encode()
        hashed_dir = hashlib.sha256(encoded).hexdigest()
        short_hash = hashed_dir[:8]
        output_dir = "test_outputs/success_rate/" + short_hash
        os.makedirs(output_dir, exist_ok=True)
        with open(f"{output_dir}/test_spec.txt", "w") as file:
            json.dump(run_spec, file, indent=4)
            file.write("\n")
            file.write("\n")
            file.write("Data Structure :\n")
            file.write("----------------\n")
            file.write(df_results.dtypes.to_string()) 
    
        df_results.to_pickle(f"{output_dir}/test_data")
        print("Done !")
    except Exception as e:
        print(f"Something went wrong: {e}")
        df_results.to_pickle(f"rescue_saving/data")
        print(f"Saved execution data in rescue_saving_tmp/data, Warning! These data will be overwrite by another failed test.")
        
if __name__ == '__main__':
    # Ensure outputs dir exists
    os.makedirs("test_outputs", exist_ok=True)
    os.makedirs("rescue_saving", exist_ok=True)
    
    # Choose which benchmark function to use (set the parameters into the benchmark functions)
    #benchmark_implementations()
    #evaluate_error_reduction()
    #benchmark_replicas_sizes()
    benchmark_success_rates()

