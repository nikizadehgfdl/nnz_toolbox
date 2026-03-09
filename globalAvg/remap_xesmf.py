##
##Script to coarsen 2D data. Based on a tool from Will Cooke and Matt Harrison used for 1/12 degree ocean model output..
##
import xarray as xr
import numpy as np
import xesmf
#import matplotlib.pyplot as plt
import netCDF4 as nc
import glob
import argparse
import os
import shutil
import multiprocessing
import socket
import time
import psutil, os

def print_mem(label=''):
    p = psutil.Process(os.getpid())
    rss = p.memory_info().rss
    # include children
    for c in p.children(recursive=True):
        rss += c.memory_info().rss
    print(f"{label} RSS memory: ", rss/1024./1024., "MB")
    return rss

def cell_methods(dA):
    try:
        cm=dA.cell_methods
    except:
        return ''

    if cm is not None:
        cm_ = cm.split(' ')
        for c in cm_:
            if c.find('area')>-1:
                ca=c
                exit
            if c.find('sum')>-1:
                ca=c
                exit
        try:
            cam=ca.split(':')[1]
        except:
            return ''
    else:
        return ''

    return cam

def stagger(dA):
    dims_=dA.dims
    if 'xh' in dims_ and 'yh' in dims_:
        return 'h'
    elif 'xq' in dims_ and 'yh' in dims_:
        return 'Cu'
    elif 'xh' in dims_ and 'yq' in dims_:
        return 'Cv'
    elif 'xq' in dims_ and 'yq' in dims_:
        return 'q'
    else:
        return None

def sum_to_scalar(da):
    """Return the sum of a DataArray as a Python scalar, computing dask if necessary."""
    s = da.sum()
    data = getattr(s, 'data', s)
    if hasattr(data, 'compute'):
        data = data.compute()
    arr = np.asarray(data)
    try:
        return arr.item()
    except Exception:
        return float(arr)

def save_dataset_dask(ds, path, unlimited_dims=['time']):
    """Write Dataset via xarray+dask: to_netcdf(compute=False) then dask.compute."""
    print(f"Saving dataset via dask to {path}")
    try:
        delayed = ds.to_netcdf(path, mode='w', unlimited_dims=unlimited_dims, compute=False)
    except TypeError:
        # backend doesn't support delayed write
        print("Backend doesn't support delayed write; performing direct write")
        return ds.to_netcdf(path, unlimited_dims=unlimited_dims)

    try:
        import dask
        dask.compute(delayed)
    except Exception as e:
        print("Dask compute failed, falling back to direct write:", e)
        return ds.to_netcdf(path, unlimited_dims=unlimited_dims)

def prepare_input_grid_for_xesmf(hgrid_file):
    hgrid = xr.open_dataset(hgrid_file)
    lon = hgrid['x'].values[1::2,1::2].copy()
    lon_bnds = hgrid['x'].values[0::2,0::2].copy()
    lat = hgrid['y'].values[1::2,1::2].copy()
    lat_bnds = hgrid['y'].values[0::2,0::2].copy()
    grid = xr.Dataset()
    grid['lon'] = xr.DataArray(data=lon, dims=('yh','xh'))
    grid['lon_b'] = xr.DataArray(data=lon_bnds, dims=('yq','xq'))
    grid['lat'] = xr.DataArray(data=lat, dims=('yh', 'xh'))
    grid['lat_b'] = xr.DataArray(data=lat_bnds, dims=('yq', 'xq'))
    return grid

def prepare_input_grid_model_for_xesmf_static(ds):
    lon2d = ds['geolon'].values
    lat2d = ds['geolat'].values
    lon_b = ds['geolon_c'].values
    lat_b = ds['geolat_c'].values
    mask  = ds['wet'].values
    grid_model = xr.Dataset()
    grid_model['lon']  = xr.DataArray(lon2d, dims=('yh','xh'))
    grid_model['lat']  = xr.DataArray(lat2d, dims=('yh','xh'))
    grid_model['lon_b'] = xr.DataArray(data=lon_b, dims=('yq','xq'))
    grid_model['lat_b'] = xr.DataArray(data=lat_b, dims=('yq','xq'))
    grid_model['mask'] = xr.DataArray(data=mask, dims=('yh','xh'))
    return grid_model

def create_target_grid_mom6(nlon,nlat):
    target_grid = xr.Dataset()
    target_grid['lon'] = xr.DataArray(data=0.5 + np.arange(nlon), dims=('x'))
    target_grid['lat'] = xr.DataArray(data=0.5 -90 + np.arange(nlat), dims=('y'))
    target_grid['lon_b'] = xr.DataArray(data=np.arange(nlon+1), dims=('xp'))
    target_grid['lat_b'] = xr.DataArray(data=-90 + np.arange(nlat+1), dims=('yp'))  
    return target_grid

def create_target_grid(nlon,nlat):
    target_lon_bnds = np.arange(-300., 61., 360./nlon)
    target_lon = target_lon_bnds[:-1] + 360./nlon/2.0

    target_lat_bnds = np.arange(-90., 91., 180./nlat)
    target_lat = target_lat_bnds[:-1] + 180./nlat/2.0

    target_grid = xr.Dataset()
    lon2d, lat2d = np.meshgrid(target_lon, target_lat)   # shape (len(target_lat), len(target_lon))
    target_grid['lon']  = xr.DataArray(lon2d, dims=('yh','xh'))
    target_grid['lat']  = xr.DataArray(lat2d, dims=('yh','xh'))

    # bounds (if you need them)
    lonb2d, latb2d = np.meshgrid(target_lon_bnds, target_lat_bnds)
    target_grid['lon_b'] = xr.DataArray(lonb2d, dims=('yq','xq'))
    target_grid['lat_b'] = xr.DataArray(latb2d, dims=('yq','xq'))
    return target_grid
    
##########################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script for implied atmospheric meridional moisture transport.''')
    parser.add_argument('-f','--inputfile', type=str, default='foo.nc', help='''input filepath ''')
    parser.add_argument('-o','--outputfile', type=str, default='foo.nc', help='''output filepath ''')
    parser.add_argument('-g','--inputgridfile', type=str, default='ocean_hgrid.nc', help='''path to input ocean grid file ''')
    parser.add_argument('-s','--staticgridfile', type=str, default='ocean_static.nc', help='''path to input static ocean grid file ''')
    parser.add_argument('--nlat', type=int, default=1, help='''Number of latitude points in target grid.''')
    parser.add_argument('--nlon', type=int, default=1, help='''Number of longitude points in target grid.''')
    parser.add_argument('--varname', type=str, nargs='+', default=['all'], help='''Name(s) of the variable(s) to coarsen. Default is "all", which means all variables will be coarsened.''')
    parser.add_argument('--dask_workers', type=int, default=1, help='''Number of Dask workers to use. Default is 1, which means no Dask parallelization.''')
    parser.add_argument('--dask_write', action='store_true', help='Use dask to perform the final to_netcdf write (to_netcdf(compute=False) then compute)')

    args = parser.parse_args()

    t1 = time.time()
    device = 'cpu'

    print_mem('Start of the script')

    n_workers = args.dask_workers
    threads_per_worker = 1
    if(args.dask_workers > 1):
        # Start Dask cluster
        from dask.distributed import Client, LocalCluster
        memory_limit = '32GB' #32G seems to be eough for 1/12th degree ocean data on PAN
        print("Starting Dask LocalCluster: workers=%s threads=%s mem=%s",
                n_workers, threads_per_worker, memory_limit)
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        client = Client(cluster)
        print("Dask client started. Dashboard: %s", client.dashboard_link)

    path=args.inputfile
    time_chunk_size=None
    if args.dask_workers >= 1:
        time_chunk_size = 1  # Adjust this based on your memory constraints and dataset size
    ds=xr.open_dataset(path, decode_times = False, chunks={'time': time_chunk_size} if args.dask_workers >= 1 else None)

    print_mem('After open_dataset calls')
    exclude_list=['uo','vo','wo'] # These will be calculated later.

    #prepare input grid  for xesmf 
    #grid0 = prepare_input_grid_for_xesmf(args.inputgridfile)
    ds_static = xr.open_dataset(args.staticgridfile)
    grid0 = prepare_input_grid_model_for_xesmf_static(ds_static)
    #create target grid for xesmf
    target_grid = create_target_grid_mom6(args.nlon, args.nlat)

    print("grid shapes,  source ",  grid0['lon'].shape, grid0['lat'].shape)
    print("grid shapes,  target ",  target_grid['lon'].shape, target_grid['lat'].shape)
    print("source lat range:", float(grid0['lat'].min()), float(grid0['lat'].max()))
    print("source lon range:", float(grid0['lon'].min()), float(grid0['lon'].max()))
    print("target lat range:", float(target_grid['lat'].min()), float(target_grid['lat'].max()))
    print("target lon range:", float(target_grid['lon'].min()), float(target_grid['lon'].max()))

    print("Making the xesmf regridder object ...")
    regridder = xesmf.Regridder(grid0, target_grid, 'conservative', periodic=True)
    ds_regrid=[]
    for nm in ds.data_vars:
        if nm in exclude_list:
            print(nm+" will be calculated later. Ignoring variable in original file.")
            continue
        if args.varname != ['all'] and nm not in args.varname:
            print(nm+" was not requested in the args. Ignoring variable in original file.")
            continue

        ds_nm = ds[nm] #to read one variable at a time and reduce memory usage. No, that doesn't do it.
        if ds_nm.ndim < 3:
            print(nm+" has less than 3 dimensions. Skipping regridding.")
            continue
        
        ntime=ds_nm.shape[0]
        #s_=stagger(ds_nm)
        #print(nm,s_, cell_methods(ds_nm))
        #Set up regridding with dask
        print("Regridding variable ",nm, " with shape ", ds_nm.shape)
        var1_regrid = regridder(ds_nm, skipna=True).compute()  # Trigger the computation and load the data into memory
        print("Regridded variable shape:", var1_regrid.shape)
        ds_regrid.append(var1_regrid.to_dataset(name=nm))

    print_mem('After remapping variables')
    # make a checksum to compare with different runs (safe for dask-backed arrays)
    checksum = 0 #sum([sum_to_scalar(d) for d in ds_regrid])
    rss = print_mem('At the end of compuations')
    t2 = time.time()
    elapsed = int(t2 - t1)
    mem_gb = rss / 1024.0 / 1024.0 / 1024.0
    print(f"Compute took {elapsed} seconds to run on host {socket.gethostname()} using device {device}, consumed memory {mem_gb:5.2f} GB, using {n_workers * threads_per_worker} dask workers, checksum: {checksum}")


    #Close Dask cluster if it was started
    if(args.dask_workers > 1):
        try:
            client.close()
            cluster.close()
        except Exception:
            pass

    time_bnds=ds['time_bnds']


    ds_out=xr.merge([xr.merge(ds_regrid),time_bnds])
    print(ds_out)

    nchunks=ntime
    chunk_size=int(ntime/nchunks)
    tstart=range(0,ntime,chunk_size)

    save_dataset_dask(ds_out, args.outputfile, unlimited_dims=['time'])

    rss = print_mem('At the end of the script')
    t2 = time.time()
    elapsed = int(t2 - t1)
    mem_gb = rss / 1024.0 / 1024.0 / 1024.0
    print(f"It took {elapsed} seconds to run on host {socket.gethostname()} using device {device}, consumed memory {mem_gb:5.2f} GB, using {n_workers * threads_per_worker} dask workers, checksum: {checksum}")

#Sample runs:
#conda activate /nbhome/ogrp/python/envs/stable
#1/4 degree ocean data:
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $ python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/remap_xesmf.py -f /xtmp/Niki.Zadeh/work/ESM4.5/ocean_monthly_z.000601-001012.thetao.nc -g /archive/gold/datasets/OM5_025/coupled_mosaic_v20250916_unpacked/ocean_hgrid.nc -o ./ESM4.5_regrid1x1.nc --nlon 360 --nlat 180 --varname thetao --dask_workers 1
#It took 80 seconds to run on host pp401 using device cpu, consumed memory  3.54 GB, using 1 dask workers, checksum: 0
#1/12 degree ocean data:
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $ python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/remap_xesmf.py -f /xtmp/Niki.Zadeh/work/William.Cooke/SPEAR/SPEAR_HI_8/SPEAR_c384_OM4p08_Control_1990_A14/01980101.ocean_z_month.nc -g /archive/wfc/gold/datasets/SPEAR/C384_OM_4500x3528/c384_mosaic_110124_expanded/ocean_hgrid.nc -o ./SPEAR12th_regrid1x1.nc --nlon 360 --nlat 180 --varname thetao --dask_workers 1
#It took  640 seconds to run on host pp401 using device cpu, consumed memory 15.97 GB, using 1 dask workers, checksum: 0
#6vars, 1/12 degree ocean data:
#It took 2293 seconds to run on host pp401 using device cpu, consumed memory 16.19 GB, using 1 dask workers, checksum: 0
#6vars, 1/12 degree ocean data:
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/remap_xesmf.py -f /ptmp/William.Cooke/archive/William.Cooke/SPEAR/SPEAR_HI_8/SPEAR_c384_OM4p08_Control_1990_A13/history/02370101.nc/02370101.ocean_z_month.nc -g /archive/wfc/gold/datasets/SPEAR/C384_OM_4500x3528/c384_mosaic_110124_expanded/ocean_hgrid.nc -o SPEAR_c3808_Control_1990_A13_1x1.nc --nlon 360 --nlat 180
#It took 4771 seconds to run on host an206 using device cpu, consumed memory 15.85 GB, using 1 dask workers, checksum: 0
#
#1/16 degree ocean data:
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $ time python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/remap_xesmf.py -f  /archive/oar.gfdl.cm5/frerts/FMS2025.03_cm5hires_08062025/cm5_hiresv2_am5f8d6r1_om5_b11_nonbous_piC/gfdl.ncrc6-intel23-prod-openmp/pp/ice/ts/monthly/5yr/ice.002601-003012.sithick.nc -s /archive/oar.gfdl.cm5/input_files/om5/OM5_0625/coupled_mosaic_v20250617_unpacked/ocean_static_nomask.nc -o ./CM5hires_ice.002601-003012.sithick_1x1_conservative_nowet.nc --nlon 360 --nlat 180 --varname sithick
#It took 420 seconds to run on host pp401 using device cpu, consumed memory 23.93 GB, using 1 dask workers, checksum: 0
#
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $  time /work/Niki.Zadeh/exchange_grid_toolset/tools/FRE-NCtools/build/src/fregrid --standard_dimension --input_mosaic /archive/oar.gfdl.cm5/input_files/om5/OM5_0625/coupled_mosaic_v20250617_unpacked/ocean_mosaic.nc --input_file ice.002601-003012.sithick --interp_method conserve_order1  --nlon 360 --nlat 180 --scalar_field sithick --output_file out_sithick.nc
#****fregrid: first order conservative scheme will be used for regridding.
#fregrid: --standard_dimension is set
#NOTE: done calculating index and weight for conservative interpolation
#Successfully running fregrid and the following output file are generated.
#****out_sithick.nc
#
#real    18m18.118s
#user    17m56.205s
#sys     0m4.430s
#
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $ time python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/remap_xesmf.py -f  /archive/oar.gfdl.cm5/frerts/FMS2025.03_cm5hires_08062025/cm5_hiresv2_am5f8d6r1_om5_b11_nonbous_piC/gfdl.ncrc6-intel23-prod-openmp/pp/ocean_monthly/ts/monthly/5yr/ocean_monthly.002601-003012.tos.nc -s /archive/oar.gfdl.cm5/input_files/om5/OM5_0625/coupled_mosaic_v20250617_unpacked/ocean_static_nomask.nc -o ./CM5hires_ocean_monthly.002601-003012.tos.1x1deg_wetmask_nan.nc --nlon 360 --nlat 180 --varname tos
#It took 420 seconds to run on host pp401 using device cpu, consumed memory 23.93 GB, using 1 dask workers, checksum: 0
# 
#Comparing with fregrid output:
#(stable) Niki.Zadeh: /xtmp/Niki.Zadeh/work/tmp $  time /work/Niki.Zadeh/exchange_grid_toolset/tools/FRE-NCtools/build/src/fregrid --standard_dimension --input_mosaic /archive/oar.gfdl.cm5/input_files/om5/OM5_0625/coupled_mosaic_v20250617_unpacked/ocean_mosaic.nc  --input_file ocean_monthly.002601-003012.tos --interp_method conserve_order1  --nlon 360 --nlat 180 --scalar_field tos --output_file out.nc
#****fregrid: first order conservative scheme will be used for regridding.
#fregrid: --standard_dimension is set
#NOTE: done calculating index and weight for conservative interpolation
#Successfully running fregrid and the following output file are generated.
#****out.nc
#
#real    18m33.927s
#user    17m47.610s
#sys     0m5.053s