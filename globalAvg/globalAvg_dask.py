#!/usr/bin/env python3
"""
Calculates global averages from postprocessed data.
Usage example:
    python ./globalAvg_torch.py 
    --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc
    --vars thetao
    --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000101-000512.volcello.nc
    --outputfile ./ESM4.5v14.000601-001012.thetao.global.nc
    --device cpu


This script will:
- Start a local Dask LocalCluster
- Load `volcello` (grid cell volume) and variables (e.g. thetao, so)
- Chunk along the time dimension and compute weighted spatial means in parallel
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
                       ncformat: str = "NETCDF3_CLASSIC"):

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
        ds_vol = xr.open_dataset(volcello_file)
        volcello = ds_vol['volcello'][-1].fillna(0)

        # total volume (scalar)
        if(use_torch):
            volcello_torch = torch.from_numpy(volcello.values).to(device)
            totalvolume = torch.sum(volcello_torch)
        else:
            totalvolume = np.sum(volcello.values)

        logger.info("Opening variable dataset: %s", inputfile)
        ds_var = xr.open_dataset(inputfile)

        for var in vars_to_process:
            var_da = ds_var[var][:].fillna(0)
            global_avg_da = []
            # Compute weighted global mean per time index
            # Use explicit multiplication to avoid potential xarray.weighted memory issues

            if(n_workers is not None): #use dask          
                var_da = var_da.chunk({tdim: time_chunk})
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
                for t in range(var_da.sizes[tdim]):
                    var_torch = torch.from_numpy(var_da[t].values).to(device)
                    weighted_sum = torch.sum(var_torch * volcello_torch)
                    global_avg_da.append(weighted_sum / totalvolume)
                global_avg[var] = xr.DataArray(torch.stack(global_avg_da).cpu().numpy(), dims=[tdim], coords={tdim: var_da[tdim]}, name=var)  
                #print("torch mean: ",global_avg[var].mean().values)
            else:
                for t in range(var_da.sizes[tdim]):
                    var_np = var_da[t].values
                    weighted_sum = np.sum(var_np * volcello.values)
                    global_avg_da.append(weighted_sum / totalvolume)
                global_avg[var] = xr.DataArray(np.stack(global_avg_da), dims=[tdim], coords={tdim: var_da[tdim]}, name=var)  
                #print("numpy mean: ",global_avg[var].mean().values)
            
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
    parser.add_argument("--mem", default="8GB", help="Memory limit per Dask worker")
    parser.add_argument("--chunk-time", type=int, default=1, help="Time chunk size for processing")   
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
                       ncformat=args.ncformat)


if __name__ == '__main__':
    main()


#command line example and timings:
#PAN an206:
#module load conda;conda activate /nbhome/Niki.Zadeh/opt/miniconda3/envs/plattorch
#Dask
#python ../globalAvg/globalAvg_dask.py --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.volcello.nc --outputfile ./ESM4.5v14.000601-001012.dask.global.nc --vars thetao  --dask_workers 2 --mem 16GB --chunk-time 1
#...
#2026-01-08 15:51:01,850 INFO: Starting Dask LocalCluster: workers=2 threads=1 mem=16GB
#... 
#2026-01-08 15:52:56,977 INFO: It took 112 seconds to run on host an206 using device None, using 2 dask workers, average value: 3.545721
#Repeat with --dask_workers 6 
#2026-01-08 15:57:28,664 INFO: It took 126 seconds to run on host an206 using device None, using 6 dask workers, average value: 3.545721
#2026-01-08 15:59:52,550 INFO: It took 91 seconds to run on host an206 using device None, using 12 dask workers, average value: 3.545721
#Numpy
#python ../globalAvg/globalAvg_dask.py --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.volcello.nc --outputfile ./ESM4.5v14.000601-001012.numpy.global.nc --vars thetao
#2026-01-08 16:14:58,448 INFO: It took 38 seconds to run on host an206 using device None, average value: 3.545721
#Torch CPU
#python ../globalAvg/globalAvg_dask.py --inputfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc --staticfile /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.volcello.nc --outputfile ./ESM4.5v14.000601-001012.numpy.global.nc --vars thetao --device cpu
#2026-01-08 16:16:40,472 INFO: It took 37 seconds to run on host an206 using device cpu, average value: 3.545720
#