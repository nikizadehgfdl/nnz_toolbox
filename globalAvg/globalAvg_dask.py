#!/usr/bin/env python3
"""
Calculates global averages from postprocessed data.
Usage example:
    python ./globalAvg_dask.py 
    --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc
    --vars thetao
    --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000101-000512.volcello.nc
    --outputfile ./ESM4.5v14.000601-001012.thetao.global.nc
 either:
       --device cpu      #This can be 'cpu' or 'cuda' if torch with cuda is available, otherwise omit this argument to use numpy
 or:
       --dask_workers 4  #number of Dask workers, omit to not use Dask 
       --mem 16GB
       --chunk-time 1


This script will:
- Optionally start a local Dask LocalCluster
- If using dask, chunk along the time dimension and compute weighted spatial means in parallel
- Load `volcello` (grid cell volume) and variables (e.g. thetao, so)
- Compute weighted global means for each time index using either numpy or torch or dask
- Save the resulting time series to a NetCDF file

Requirements:
    pip install xarray dask distributed netcdf4

"""
import argparse
from pathlib import Path
import logging
import socket
import time

import xarray as xr
from dask.distributed import Client, LocalCluster
import numpy as np
import torch

logger = logging.getLogger("globalAvg_dask")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

def compute_global_avg(inputfile: str,
                       staticfile: str, 
                       vars_to_process: list,
                       xdim: str = "xh",
                       ydim: str = "yh",
                       zdim: str = "z_l",
                       tdim: str = "time",
                       n_workers: int = None,
                       threads_per_worker: int = 1,
                       memory_limit: str = "8GB",
                       time_chunk: int = 1,
                       outpath: str = None,
                       device: str = None,
                       ncformat: str = "NETCDF3_CLASSIC",
                       spatial_block_size: int = 1024):

    use_torch = True if device=="cuda" or device=="cpu" else False

    if(use_torch):
        if(device=="cuda" and not torch.cuda.is_available()):
            logger.warning("CUDA device requested but not available.")
        logger.info("Using torch with device: %s", device)
        device = torch.device(device)
    else:
        logger.info("Using numpy for computations.")

    if(n_workers is not None):
        # Start Dask cluster
        logger.info("Starting Dask LocalCluster: workers=%s threads=%s mem=%s",
                    n_workers, threads_per_worker, memory_limit)
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        client = Client(cluster)
        logger.info("Dask client started. Dashboard: %s", client.dashboard_link)


    t1 = time.time()
    global_avg = {}
    globalAvgAnnual = {}
    try:
        # Load volcello (weights)
        volcello_file = staticfile
        logger.info("Loading volcello from %s", volcello_file)
        ds_vol = xr.open_dataset(volcello_file,chunks=-1) # decode_times=False
        volcello = ds_vol['volcello'][-1].fillna(0)

        # total volume (scalar)
        if(use_torch):
            volcello_torch = torch.from_numpy(volcello.values).to(device)
            totalvolume = torch.sum(volcello_torch)
        else:
            totalvolume = np.sum(volcello.values)

        logger.info("Opening variable dataset: %s", inputfile)
        ds_var = xr.open_dataset(inputfile, chunks={tdim: time_chunk}) # decode_times=False

        for var in vars_to_process:
            # Don't load entire array at once - use lazy loading
            var_da = ds_var[var]  # Don't use [:] yet
            global_avg_da = []
            
            # Get dimension info without loading data
            xsize = var_da.sizes[xdim]
            ysize = var_da.sizes[ydim]
            ntimes = var_da.sizes[tdim]
            print("Debug: variable ", var, " sizes: time=", ntimes, ", y=", ysize, ", x=", xsize)
            # Compute weighted global mean per time index
            # Use explicit multiplication to avoid potential xarray.weighted memory issues

            if(n_workers is not None): #use dask          
                logger.info("Chunked variable %s with %s=%s", var_da.name, tdim, time_chunk)
                ##Cannot mix torch and dask: global_mean.compute() AttributeError: 'Tensor' object has no attribute 'compute'
                ##var_torch = torch.from_numpy(var_da.values).to(device)
                ##weighted_sum = torch.sum(var_torch * volcello_torch)
                weighted_sum = (var_da * volcello).sum(dim=[xdim, ydim, zdim])
                global_mean = weighted_sum / totalvolume

                logger.info("Computing global mean for %s (this will be executed in parallel)", var)
                global_avg_da = global_mean.compute()
                
                global_avg[var] = xr.DataArray(global_avg_da.values, dims=[tdim], coords={tdim: var_da[tdim]}, name=var)               
                #print("dask mean: ",global_avg[var].mean().values)
            
            elif(use_torch):
                # Process in time chunks to fit memory
                dim_indices = [list(var_da.dims).index(d) for d in [xdim, ydim, zdim] if d in var_da.dims] # gives [3, 2, 1]
                chunk_time_size = min(time_chunk, ntimes)
                for t_start in range(0, ntimes, chunk_time_size):
                    print("Processing time chunks: ", t_start, " to ", min(t_start + chunk_time_size, ntimes), " of ", ntimes)
                    t_end = min(t_start + chunk_time_size, ntimes)
                    # Load only this time chunk
                    var_chunk = var_da.isel({tdim: slice(t_start, t_end)}).fillna(0).values
                    #var_chunk[np.isnan(var_chunk)] = 0 #Instead of .fillna(0) above, but is slower
                    varchunk_GB = var_chunk.nbytes / (1024**3)
                    print(f"Variable {var} chunk size in GB: {varchunk_GB:.2f} GB, shape: {var_chunk.shape}")
                    if varchunk_GB > 35:
                        print("Chunk too big to fit in memory, reduce time_chunk size")
                        break
                    var_torch = torch.from_numpy(var_chunk).to(device)
                    #var_torch[torch.isnan(var_torch)] = 0 #Instead of .fillna(0) above, but tries to allocate additional memory on GPU and cause crash
                    var_torch.mul_(volcello_torch) #use in-place multiplication to save memory
                    weighted_sum = torch.sum(var_torch, dim=dim_indices)
                    global_avg_da.append(weighted_sum / totalvolume)
                    del var_torch
                global_avg[var] = xr.DataArray(torch.cat(global_avg_da).cpu().numpy(), dims=[tdim], coords={tdim: var_da[tdim]}, name=var)
                #global_avg[var] = xr.DataArray(np.array(global_avg_da), dims=[tdim], coords={tdim: var_da[tdim]}, name=var)  
            else:
                dim_indices = tuple(var_da.dims.index(d) for d in [xdim, ydim, zdim] if d in var_da.dims)
                chunk_time_size = min(time_chunk, ntimes)
                for t_start in range(0, ntimes, chunk_time_size):
                    print("Processing time chunks: ", t_start, " to ", min(t_start + chunk_time_size, ntimes), " of ", ntimes)
                    t_end = min(t_start + chunk_time_size, ntimes)
                    # Load only this time chunk
                    var_chunk = var_da.isel({tdim: slice(t_start, t_end)}).fillna(0).values
                    weighted_sum = np.sum(var_chunk * volcello.values, axis=dim_indices)
                    # Flatten to ensure 1D per time step
                    global_avg_da.extend(weighted_sum.flatten() / totalvolume)
                global_avg[var] = xr.DataArray(np.array(global_avg_da), dims=[tdim], coords={tdim: var_da[tdim]}, name=var)  
            
            #Compute annual means                
            globalAvgAnnual[var] = global_avg[var].resample(time='AS').mean(dim='time')

        # Combine into a Dataset and save
        out_path = outpath or (Path(ppdir) / "global_avg_monthly.nc")
        logger.info("Saving results to %s", out_path)
        ds_out = xr.Dataset({v: global_avg[v] for v in global_avg})
        # Save annual datase
        out_path2 = outpath[:-3] + "_annual.nc"
        logger.info("Saving results to %s", out_path2)
        ds_out2 = xr.Dataset({v: globalAvgAnnual[v] for v in globalAvgAnnual})

        # Ensure time coordinate name is preserved; if variables have different time coords, xarray will align
        # Write using requested netCDF format (e.g. NETCDF3_CLASSIC, NETCDF4, NETCDF4_CLASSIC)
        encoding = {v: {} for v in global_avg}  # per-variable encoding
        encoding['time'] = {'unlimited': True}
        logger.info("Writing output to %s using format=%s", out_path, ncformat)
        ds_out.to_netcdf(str(out_path), format=ncformat, unlimited_dims=['time'])
        logger.info("Saved output to %s", out_path)
        logger.info("Writing output to %s using format=%s", out_path2, ncformat)
        ds_out2.to_netcdf(str(out_path2), format=ncformat, unlimited_dims=['time'])
        logger.info("Saved output to %s", out_path2)

    finally:
        if(n_workers is not None):
            logger.info("Closing Dask client and cluster")
            try:
                client.close()
                cluster.close()
            except Exception:
                pass

    t2 = time.time()
    if(n_workers is not None):
        logger.info("It took %d seconds to run on host %s using device %s, using %d dask workers, average value: %f", t2-t1, str(socket.gethostname()), device, n_workers * threads_per_worker, global_avg[var].mean().values)
    else:
        logger.info("It took %d seconds to run on host %s using device %s, average value: %f", t2-t1, str(socket.gethostname()), device, global_avg[var].mean().values)

    return out_path


def main():
    parser = argparse.ArgumentParser(description="Compute weighted global means for given variables using Dask")
    parser.add_argument("--inputfile", required=True, help="Path to the input NetCDF file containing the variable data to process")
    parser.add_argument("--staticfile", required=True, help="Path to the static NetCDF file containing volcello data")
    parser.add_argument("--outputfile", required=True, help="Path to the output NetCDF file to save the global averages")
    parser.add_argument("--vars", required=True, nargs='+', default=["thetao"], help="Variables to process (space-separated)")
    parser.add_argument("--ncformat", default="NETCDF3_CLASSIC", help="netCDF format to write: NETCDF3_CLASSIC, NETCDF4, NETCDF4_CLASSIC")
    parser.add_argument("--device", default=None, help="Device to use: 'cpu' or 'cuda'")
    parser.add_argument("--dask_workers", type=int, default=None, help="Number of Dask workers")
    parser.add_argument("--mem", default="16GB", help="Memory limit per Dask worker")
    parser.add_argument("--chunk_time", type=int, default=1, help="Time chunk size for processing")   
    parser.add_argument("--spatial_block_size", type=int, default=1024, help="Spatial block size for processing")   
    args = parser.parse_args()

    if(args.device=="cuda" and not torch.cuda.is_available()):
        print("CUDA device requested but not available.")

    compute_global_avg(inputfile=args.inputfile,
                       staticfile=args.staticfile,
                       vars_to_process=args.vars,
                       n_workers=args.dask_workers,
                       memory_limit=args.mem,
                       time_chunk=args.chunk_time,
                       outpath=args.outputfile,
                       device=args.device,
                       ncformat=args.ncformat,
                       spatial_block_size=args.spatial_block_size)


if __name__ == '__main__':
    main()


#command line example and timings:
#small and fast test
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/globalAvg_dask.py --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.volcello.nc --outputfile ./ESM4.5v14.000601-001012.numpy.global.nc --vars thetao --device cpu --chunk_time 10
#2026-01-28 12:07:08,679 INFO: It took 41 seconds to run on host pp401 using device cpu, average value: 3.545720
#2026-01-28 12:08:44,431 INFO: It took 29 seconds to run on host pp401 using device cuda, average value: 3.545720
#2026-01-28 12:37:01,820 INFO: It took 47 seconds to run on host pp401 using device None, average value: 3.545721
#
#Large and slow test
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/globalAvg_dask.py --inputfile ./ocean_monthly_z.000101-000512.thetao.nc --staticfile ./ocean_monthly_z.000101-000512.volcello.nc --outputfile ./CM5hires.global.nc --vars thetao --dask_workers 10
#
### files on /work
##dask
#2026-01-28 09:05:50,297 INFO: It took 598 seconds to run on host pp401 using device None, using 2 dask workers, average value: 3.569357
#2026-01-28 08:53:03,875 INFO: It took 314 seconds to run on host pp401 using device None, using 10 dask workers, average value: 3.569357
#torch and numpy
#2026-01-28 15:19:36,819 INFO: It took 996 seconds to run on host pp401 using device None, average value: 3.569357
#2026-01-28 15:38:39,663 INFO: It took 1017 seconds to run on host pp401 using device cpu, average value: 3.569368
#2026-01-28 16:05:36,604 INFO: It took 1081 seconds to run on host pp401 using device cuda, average value: 3.569368
###files on /xtmp
##dask
#2026-01-28 09:13:00,248 INFO: It took 367 seconds to run on host pp401 using device None, using 2 dask workers, average value: 3.569357
#2026-01-28 09:34:10,901 INFO: It took 219 seconds to run on host pp401 using device None, using 10 dask workers, average value: 3.569357
#2026-01-28 11:13:09,348 INFO: It took 227 seconds to run on host pp401 using device None, using 12 dask workers, average value: 3.569357
#2026-01-28 11:03:08,702 INFO: It took 224 seconds to run on host pp401 using device None, using 20 dask workers, average value: 3.569357
#torch and numpy
#2026-01-28 12:47:37,782 INFO: It took 506 seconds to run on host pp401 using device cuda, average value: 3.569368
#2026-01-28 13:08:57,732 INFO: It took 432 seconds to run on host pp401 using device cuda, average value: nan without .fillna(0)
#2026-01-28 14:26:14,552 INFO: It took 529 seconds to run on host pp401 using device cuda, average value: 3.569368
#2026-01-28 14:46:06,630 INFO: It took 600 seconds to run on host pp401 using device cpu, average value: 3.569368
#2026-01-28 15:01:40,458 INFO: It took 673 seconds to run on host pp401 using device None, average value: 3.569357