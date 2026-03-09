##
##Script to coarsen 2D data. Based on a tool from Will Cooke and Matt Harrison used for 1/12 degree ocean model output..
##
import xarray as xr
import numpy as np
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

def block_coarsen_da_yx(da_xr, var_name, nagg, stagger='h', how='mean', time_chunk_size=1):
    """Fast block coarsen using dask coarsen. 
       This gives the same answers as block_reshape_da_yx and has a similar runtime and memory usage.
       This must be called after chunking the input DataArray by time, otherwise the reshape will not work
       because the data will be a numpy array instead of a dask array. 
       The output is computed at the end to return a concrete DataArray, 
       but the intermediate steps are lazy and should not consume much memory. 
       The reshape is done in y,x order to be more cache-friendly for Fortran-style arrays. 
       The stagger argument specifies the grid staggering of the input variable, which determines how the coarsening is done.
    """
    import dask.array as da

    print(f"block_coarsen_da_yx: varname={var_name}, how={how}, nagg={nagg}")
    #
    var = da_xr.data #lazy
    #if not hasattr(var, 'chunks'):
    #    raise ValueError("Input DataArray must be chunked by time before calling block_reshape_da_yx, use --dask_workers > 0.")
    #    It will run, but slow

    ntime = var.shape[0]
    nz = var.shape[1]
    nlat = var.shape[2]
    nlon = var.shape[3]
    nlat_coarse = nlat // nagg
    nlon_coarse = nlon // nagg

    # Build new coordinates to return a proper DataArray.
    coords = {}
    coords['time'] = da_xr['time'].values[:ntime]
    coords['z_l'] = da_xr['z_l'].values[:nz]

    if stagger == 'h':
        if how == 'sum':
            var_coarse =  da.coarsen(np.sum, var, {2: nagg, 3: nagg}, trim_excess=True)
        else:
            #want  : da_xr.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean()
            #faster: da_xr.coarsen(yh=nagg).mean().coarsen(xh=nagg).mean()
            var_coarse =  da.coarsen(np.mean, var, {2: nagg, 3: nagg}, trim_excess=True)

        coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)
        coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)
        dims = ['time', 'z_l', 'yh', 'xh']
    elif stagger == 'Cu':
        if how == 'sum':
            #want:  da_xr[:,:,:,::args.nagg].coarsen(yh=args.nagg).sum()
            #var_coarse = var[:,:,:,::nagg]
            #var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nagg, nlon_coarse).sum(axis=3).compute()
            ##faster if flip x,y order
            var_coarse = da.coarsen(np.sum, var, {2: nagg}, trim_excess=True)
            var_coarse = var_coarse[:,:,:,::nagg]
            #print('var_coarse.shape:', var_coarse.shape)
        else:
            var_coarse = da.coarsen(np.mean, var, {2: nagg}, trim_excess=True)
            var_coarse = var_coarse[:,:,:,::nagg]

        coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)
        coords['xq'] = da_xr['xq'].values[::nagg]
        dims = ['time', 'z_l', 'yh', 'xq']
    elif stagger == 'Cv':
        if how == 'sum':
            #want: da_xr[:,:,::args.nagg,:].coarsen(xh=args.nagg).sum()
            #faster if flip x,y order
            var_coarse = da.coarsen(np.sum, var, {3: nagg}, trim_excess=True)
            var_coarse = var_coarse[:,:,::nagg,:]
        else:
            var_coarse = da.coarsen(np.mean, var, {3: nagg}, trim_excess=True)
            var_coarse = var_coarse[:,:,::nagg,:]

        coords['yq'] = da_xr['yq'].values[::nagg]
        coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)
        dims = ['time', 'z_l', 'yq', 'xh']

    da_coarse = xr.DataArray(var_coarse, dims=dims, coords=coords, attrs=da_xr.attrs, name=var_name)
    return da_coarse.compute()

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

def save_timeslice(ds,i,n,ntime):
    fout_tmp='tmpdir_ocean_month_z/om_'+str(i).zfill(4)+'.nc'
    print('process id:', os.getpid(),fout_tmp)
    ds.isel(time=slice(i,min(i+n,ntime))).to_netcdf(fout_tmp,unlimited_dims=['time',],compute=True)

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
    
##########################    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script for implied atmospheric meridional moisture transport.''')
    parser.add_argument('-f','--inputfile', type=str, default='foo.nc', help='''input filepath ''')
    parser.add_argument('-n','--nagg', type=int, default=3, help='''year ''')
    parser.add_argument('--varname', type=str, nargs='+', default=['all'], help='''Name(s) of the variable(s) to coarsen. Default is "all", which means all variables will be coarsened.''')
    parser.add_argument('--uname', type=str, default='umo', help='''Name of the u-component of mass transport''')
    parser.add_argument('--vname', type=str, default='vmo', help='''Name of the v-component of mass transport''')
    parser.add_argument('--wrapx', type=bool, default=True, help='''True if the x-component is reentrant''')
    parser.add_argument('--wrapy', type=bool, default=False, help='''True if the y-component is reentrant''')
    parser.add_argument('--dask_workers', type=int, default=1, help='''Number of Dask workers to use. Default is 1, which means no Dask parallelization.''')
    parser.add_argument('--dask_write', action='store_true', help='Use dask to perform the final to_netcdf write (to_netcdf(compute=False) then compute)')
    parser.add_argument('-o','--outputfile', type=str, default='foo.nc', help='''output filepath ''')

    args = parser.parse_args()

    t1 = time.time()
    device = 'cpu'

    if not os.path.isdir('tmpdir_ocean_month_z'):
        os.mkdir('tmpdir_ocean_month_z')

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
    dA_i=[];dA_e=[];
    #dA_i_=[];dA_e_=[] #unused

    print_mem('After open_dataset calls')
    exclude_list=['uo','vo','wo'] # These will be calculated later.

    ntime=ds['thetao'].shape[0]
    for nm in ds.data_vars:
        if nm in exclude_list:
            print(nm+" will be calculated later. Ignoring variable in original file.")
            continue
        if args.varname != ['all'] and nm not in args.varname:
            print(nm+" was not requested in the args. Ignoring variable in original file.")
            continue

        ds_nm = ds[nm] #to read one variable at a time and reduce memory usage. No, that doesn't do it.

        s_=stagger(ds_nm)
        print(nm,s_, cell_methods(ds_nm))
        if cell_methods(ds_nm).find('sum')>-1:
            #dA_e_.append(ds_nm)
            if s_ == 'h':
                #dA_e.append(ds_nm.coarsen(xh=args.nagg).sum().coarsen(yh=args.nagg).sum()) #original, slow
                dA_e.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, how='sum'))  #dask coarsen
            elif s_ == 'Cu':
                #dA_e.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).sum()) #original, slow
                #dA_e.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='sum')) #dask reshape
                dA_e.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='sum')) #dask coarsen
            elif s_ == 'Cv':
                #dA_e.append(ds_nm[:,:,::args.nagg,:].coarsen(xh=args.nagg).sum()) #original, slow
                #dA_e.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='sum')) #dask reshape
                dA_e.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='sum')) #dask coarsen     
            elif s_ == 'q':
                print(nm,' at corners')
        elif cell_methods(ds_nm).find('mean')>-1:
            #dA_i_.append(ds_nm)
            if s_ == 'h':
                #dA_i.append(ds_nm.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean()) #original, slow
                #dA_i.append(ds_nm.coarsen(yh=args.nagg).mean().coarsen(xh=args.nagg).mean()) #original xy flip, faster
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='h', how='mean')) #dask coarsen
            elif s_ == 'Cu':
                #dA_i.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='mean'))
            elif s_ == 'Cv':
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='mean'))
            elif s_ == 'q':
                dA_i.append(ds_nm[:,:,::args.nagg,::args.nagg])
        else:
            print(nm,' has no cell method or an unrecognized cell method. Treating as mean.')
            #dA_i_.append(ds_nm)
            if s_ == 'h':
                #dA_i.append(ds_nm.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean())
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='h', how='mean'))
            elif s_ == 'Cu':
                #dA_i.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='mean'))
            elif s_ == 'Cv':
                #dA_i.append(ds_nm[:,:,::args.nagg,:].coarsen(xh=args.nagg).mean())
                dA_i.append(block_coarsen_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='mean'))
            elif s_ == 'q':
                dA_i.append(ds_nm[:,:,::args.nagg,::args.nagg])

        #release the memory used by ds_nm
        del ds_nm

    print_mem('After computing coarsened variables')
    # make a checksum to compare with different runs (safe for dask-backed arrays)
    checksum = sum([sum_to_scalar(d) for d in dA_i]) + sum([sum_to_scalar(d) for d in dA_e])
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


    ds_out=xr.merge([xr.merge(dA_e),xr.merge(dA_i),time_bnds])

    nchunks=ntime
    chunk_size=int(ntime/nchunks)
    tstart=range(0,ntime,chunk_size)

    serial_write = False  #Use mutliprocesses to write chunks in parallel, or write serially in a loop. Parallel writes can be faster but may cause more memory usage and I/O contention, especially if the output file is on a network filesystem. 
                          #Serial writes are simpler and may be more stable for large files, but can be slower. 
                          # You can experiment with both options to see which works better for your use case and system configuration.

    if args.dask_write:
        # single-file write via dask (parallel if a Dask client is active)
        save_dataset_dask(ds_out, args.outputfile, unlimited_dims=['time'])
    else:
        if serial_write:
            # Serial writes (no child processes)
            for n in np.arange(nchunks):
                print(f"Writing chunk {n} starting at time index {tstart[n]}", flush=True)
                try:
                    save_timeslice(ds_out, tstart[n], chunk_size, ntime)
                except Exception as e:
                    print(f"Error writing chunk {n}: {e}", flush=True)
            print('Serial write complete.')
        else:
            p=[]
            for n in np.arange(nchunks):
                p.append(multiprocessing.Process(target=save_timeslice,args=(ds_out,tstart[n],chunk_size,ntime)))
            print("Deubg: Here1")
            for p_ in p:
                p_.start()
            print("Deubg: Here2")
            for p_ in p:
                p_.join()
            print(' Multicore write process complete.')

    checksum = sum([sum_to_scalar(d) for d in dA_i]) + sum([sum_to_scalar(d) for d in dA_e])

    rss = print_mem('At the end of the script')
    t2 = time.time()
    elapsed = int(t2 - t1)
    mem_gb = rss / 1024.0 / 1024.0 / 1024.0
    print(f"It took {elapsed} seconds to run on host {socket.gethostname()} using device {device}, consumed memory {mem_gb:5.2f} GB, using {n_workers * threads_per_worker} dask workers, checksum: {checksum}")

#Sample runs:
#02242026
#ESM4.5, 1/4 degree ocean
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/downSample_dask.py -f  /archive/Niki.Zadeh/CMIP7/ESM4/DEV/ESM4.5v14_nonsymmetric/gfdl.ncrc5-inteloneapi252-prod-openmp/pp/ocean_monthly_z/ts/monthly/5yr/ocean_monthly_z.000601-001012.thetao.nc --nagg 3 --varname thetao --dask_workers 1 --dask_write -o ./ESM4.5v14_downsample3_thetao.nc
#Compute took 26 seconds to run on host pp401 using device cpu, consumed memory  1.91 GB, using 1 dask workers, checksum: 910480576.0
#
#cm5hires 1/16 degree ocean
#(plattorch) Niki.Zadeh: /xtmp/Niki.Zadeh/work/hiresCM5 $  python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/downSample_dask.py -f ocean_monthly_z.000101-000512.thetao.nc --nagg 4 --varname thetao --dask_workers 1 --dask_write -o CM5hires_downsample4_thetao.nc
#Compute took 363 seconds to run on host pp401 using device cpu, consumed memory 13.37 GB, using 1 dask workers, checksum: 8356301312.0
#It took 389 seconds to run on host pp401 using device cpu, consumed memory 13.40 GB, using 1 dask workers, checksum: 8356301312.0
#Compute took 382 seconds to run on host pp401 using device cpu, consumed memory 13.65 GB, using 2 dask workers, checksum: 8356301312.0
#Compute took 241 seconds to run on host pp401 using device cpu, consumed memory 14.31 GB, using 6 dask workers, checksum: 8356301312.0
#Compute took 231 seconds to run on host pp401 using device cpu, consumed memory 15.19 GB, using 10 dask workers, checksum: 8356301312.0
#Compute took 217 seconds to run on host pp401 using device cpu, consumed memory 17.11 GB, using 20 dask workers, checksum: 8356301312.0
#
#12th degree ocean
#(plattorch) Niki.Zadeh: /xtmp/Niki.Zadeh/work/William.Cooke/SPEAR/SPEAR_HI_8/SPEAR_c384_OM4p08_Control_1990_A14 $ python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/downSample_dask.py -f 01980101.ocean_z_month.nc --nagg 3 --dask_workers 2 --varname thetao
#Compute took 125 seconds to run on host pp401 using device cpu, consumed memory  6.54 GB, using 2 dask workers, checksum: 5668890624.0
#It took 135 seconds to run on host pp401 using device cpu, consumed memory  6.13 GB, using 2 dask workers, checksum: 5668890624.0
#6vars
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/downSample_dask.py -f 01980101.ocean_z_month.nc --nagg 3 --dask_workers 6 --dask_write
#Compute took 486 seconds to run on host pp401 using device cpu, consumed memory 37.00 GB, using 6 dask workers, checksum: 2710719237373888.0
#It took 557 seconds to run on host pp401 using device cpu, consumed memory 35.73 GB, using 6 dask workers, checksum: 2710719237373888.0
#
#python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/downSample_dask.py -f 01980101.ocean_z_month.nc --nagg 3 --dask_workers 6 --dask_write -o 01980101.ocean_z_month_aggx3.nc
#Compute took 648 seconds to run on host an206 using device cpu, consumed memory 36.89 GB, using 6 dask workers, checksum: 2710719237373888.0
#It took 751 seconds to run on host an206 using device cpu, consumed memory 35.71 GB, using 6 dask workers, checksum: 2710719237373888.0
