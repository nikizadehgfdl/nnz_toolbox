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

def block_coarsen_xy(da_xr, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using xr coarsen.
    """
    print(f"block_coarsen_xy: how={how}, nagg={nagg}")
    if how == 'sum':
        return da_xr.coarsen(xh=nagg).sum().coarsen(yh=nagg).sum().compute()
    else:
        return da_xr.coarsen(xh=nagg).mean().coarsen(yh=nagg).mean().compute()

def block_coarsen_yx(da_xr, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using xr coarsen.
    """
    print(f"block_coarsen_yx: how={how}, nagg={nagg}")
    if how == 'sum':
        return da_xr.coarsen(yh=nagg).sum().coarsen(xh=nagg).sum().compute()
    else:
        return da_xr.coarsen(yh=nagg).mean().coarsen(xh=nagg).mean().compute()
    
def block_reshape_da_yx(da_xr, var_name, nagg, stagger='h', how='mean', time_chunk_size=1):                                                             
    """Fast block coarsen using numpy/dask reshape.                                                 
    """                                                                                                                 
    print(f"block_reshape_da_yx: varname={var_name}, how={how}, nagg={nagg}")                                                                  
    #                                                                                             
    var = da_xr.chunk({'time': time_chunk_size}).data #lazy                                                                           
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
            var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon).sum(axis=3)                                        
            var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).sum(axis=4).compute()                
        else:   
            #want  : da_xr.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean()                                                                                                            
            #faster: da_xr.coarsen(yh=nagg).mean().coarsen(xh=nagg).mean()
            var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon).mean(axis=3)                                       
            var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).mean(axis=4).compute()               

        coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)                                    
        coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)                                                                                                                                                            
        dims = ['time', 'z_l', 'yh', 'xh']                                                                                  
    elif stagger == 'Cu':
        if how == 'sum':  
            #want:  da_xr[:,:,:,::args.nagg].coarsen(yh=args.nagg).sum()
            #var_coarse = var[:,:,:,::nagg]
            var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon_coarse).sum(axis=3).compute()
            var_coarse = var_coarse[:,:,:,::nagg]
            #print('var_coarse.shape:', var_coarse.shape)
        else:                                                                                                               
            var_coarse = var[:,:,:,::nagg]
            var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nagg, nlon_coarse).mean(axis=3).compute()

        coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)                                    
        coords['xq'] = da_xr['xq'].values[::nagg]                                                                                                                                                            
        dims = ['time', 'z_l', 'yh', 'xq']                                                                                  
    elif stagger == 'Cv':
        if how == 'sum':  
            #want: da_xr[:,:,::args.nagg,:].coarsen(xh=args.nagg).sum()
            var_coarse = var[:,:,::nagg,:]
            var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).sum(axis=4).compute()
        else:                                                                                                               
            var_coarse = var[:,:,::nagg,:]
            var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).mean(axis=4).compute()

        coords['yq'] = da_xr['yq'].values[::nagg]                                    
        coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)                                                                                                                                                            
        dims = ['time', 'z_l', 'yq', 'xh']                                                                                  
                                                                                                                                    
    da_coarse = xr.DataArray(var_coarse, dims=dims, coords=coords, attrs=da_xr.attrs, name=var_name)                    
    return da_coarse                                                                                                    
                                                                                                                        
def block_reshape_np_yx(da_xr, var_name, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using numpy reshape.
    """
    print(f"block_reshape_np: how={how}, nagg={nagg}")
    ntime = da_xr.shape[0]
    nz = da_xr.shape[1]
    nlat = da_xr.shape[2]
    nlon = da_xr.shape[3]
    nlat_coarse = nlat // nagg
    nlon_coarse = nlon // nagg
    var = da_xr.values
    if how == 'sum':
        var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon).sum(axis=3)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).sum(axis=4)
    else:
        var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon).mean(axis=3)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).mean(axis=4)

    # Build new coordinates to return a proper DataArray. This is a bit tricky because we need to trim the original coordinates to match the coarsened dimensions, and then take the mean of the coordinates within each block to get the new coordinate values. We also need to make sure to keep the time and z_l coordinates unchanged.
    coords = {}
    coords['time'] = da_xr['time'].values[:ntime]
    coords['z_l'] = da_xr['z_l'].values[:nz]
    # Block average coordinates for yh and xh
    coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)
    coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)

    dims = ['time', 'z_l', 'yh', 'xh']
    da_coarse = xr.DataArray(var_coarse, dims=dims, coords=coords, attrs=da_xr.attrs, name=var_name)
    return da_coarse

def block_reshape_np_xy(da_xr, var_name, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using numpy reshape.
    """
    print(f"block_reshape_np: how={how}, nagg={nagg}")
    ntime = da_xr.shape[0]
    nz = da_xr.shape[1]
    nlat = da_xr.shape[2]
    nlon = da_xr.shape[3]
    nlat_coarse = nlat // nagg
    nlon_coarse = nlon // nagg
    var = da_xr.values
    if how == 'sum':
        var_coarse = var.reshape(ntime, nz, nlat, nlon_coarse, nagg).sum(axis=4)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nagg, nlon).sum(axis=3)
    else:
        var_coarse = var.reshape(ntime, nz, nlat, nlon_coarse, nagg).mean(axis=4)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nagg, nlon_coarse).mean(axis=3)
        #_yx
        #var_coarse = var.reshape(ntime, nz, nlat_coarse, nagg, nlon).mean(axis=3)
        #var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).mean(axis=4)

    # Build new coordinates to return a proper DataArray. This is a bit tricky because we need to trim the original coordinates to match the coarsened dimensions, and then take the mean of the coordinates within each block to get the new coordinate values. We also need to make sure to keep the time and z_l coordinates unchanged.
    coords = {}
    coords['time'] = da_xr['time'].values[:ntime]
    coords['z_l'] = da_xr['z_l'].values[:nz]
    # Block average coordinates for yh and xh
    coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)
    coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)

    dims = ['time', 'z_l', 'yh', 'xh']
    da_coarse = xr.DataArray(var_coarse, dims=dims, coords=coords, attrs=da_xr.attrs, name=var_name)
    return da_coarse

def block_reshape_torch_yx(da_xr, var_name, nagg, how='mean',device='cpu'):
    """Fast block coarsen for 'h' staggered fields using torch tensor reshape.
    """
    import torch
    device = torch.device(device) 

    print(f"block_reshape_torch_yx: how={how}, nagg={nagg}, device={device}")
    ntime = da_xr.shape[0]
    nz = da_xr.shape[1]
    nlat = da_xr.shape[2]
    nlon = da_xr.shape[3]
    nlat_coarse = nlat // nagg
    nlon_coarse = nlon // nagg
    var = da_xr.values
    
    var_torch = torch.from_numpy(var).to(device)

    if how == 'sum':
        var_coarse = var_torch.reshape(ntime, nz, nlat_coarse, nagg, nlon).sum(dim=3)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).sum(dim=4)
    else:
        var_coarse = var_torch.reshape(ntime, nz, nlat_coarse, nagg, nlon).mean(dim=3)
        var_coarse = var_coarse.reshape(ntime, nz, nlat_coarse, nlon_coarse, nagg).mean(dim=4)

    # Build new coordinates to return a proper DataArray. This is a bit tricky because we need to trim the original coordinates to match the coarsened dimensions, and then take the mean of the coordinates within each block to get the new coordinate values. We also need to make sure to keep the time and z_l coordinates unchanged.
    coords = {}
    coords['time'] = da_xr['time'].values[:ntime]
    coords['z_l'] = da_xr['z_l'].values[:nz]
    # Block average coordinates for yh and xh
    coords['yh'] = da_xr['yh'].values[:nlat].reshape(nlat_coarse, nagg).mean(axis=1)
    coords['xh'] = da_xr['xh'].values[:nlon].reshape(nlon_coarse, nagg).mean(axis=1)

    dims = ['time', 'z_l', 'yh', 'xh']
    da_coarse = xr.DataArray(var_coarse.cpu().numpy(), dims=dims, coords=coords, attrs=da_xr.attrs, name=var_name)
    return da_coarse

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script for implied atmospheric meridional moisture transport.''')
    parser.add_argument('-f','--filename', type=str, default='foo.nc', help='''filename ''')
    parser.add_argument('--staticFile', type=str, default='ocean_static_aggx3.nc', help='''filename ''')
    parser.add_argument('-n','--nagg', type=int, default=3, help='''year ''')
    parser.add_argument('--varname', type=str, nargs='+', default=['all'], help='''Name(s) of the variable(s) to coarsen. Default is "all", which means all variables will be coarsened.''')
    parser.add_argument('--uname', type=str, default='umo', help='''Name of the u-component of mass transport''')
    parser.add_argument('--vname', type=str, default='vmo', help='''Name of the v-component of mass transport''')
    parser.add_argument('--wrapx', type=bool, default=True, help='''True if the x-component is reentrant''')
    parser.add_argument('--wrapy', type=bool, default=False, help='''True if the y-component is reentrant''')
    parser.add_argument('--dask_workers', type=int, default=0, help='''Number of Dask workers to use. Default is 1, which means no Dask parallelization.''')

    args = parser.parse_args()
 
    t1 = time.time()
    device = 'cpu'

    if not os.path.isdir('tmpdir_ocean_month_z'):
        os.mkdir('tmpdir_ocean_month_z')

    print_mem('Before Dask setup')

    n_workers = args.dask_workers
    threads_per_worker = 1
    if(args.dask_workers > 1):
        # Start Dask cluster
        from dask.distributed import Client, LocalCluster
        memory_limit = '16GB'
        print("Starting Dask LocalCluster: workers=%s threads=%s mem=%s",
                n_workers, threads_per_worker, memory_limit)
        cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
        client = Client(cluster)
        print("Dask client started. Dashboard: %s", client.dashboard_link)

    #out_dir='aggx'+str(args.nagg)
    out_dir='.'
    fsplit=args.filename.split(sep='.')
    path_out=out_dir+'/'+fsplit[0]+'.ocean_z_month_aggx'+str(args.nagg)+'.nc'
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    path=args.filename
    time_chunk_size=None
    if args.dask_workers >= 1:
        time_chunk_size = 1  # Adjust this based on your memory constraints and dataset size
    ds=xr.open_dataset(path, decode_times = False, chunks={'time': time_chunk_size} if args.dask_workers >= 1 else None)
    static_path=args.staticFile
    ds_static=xr.open_dataset(static_path)
    dA_i=[];dA_e=[]; 
    #dA_i_=[];dA_e_=[]

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
                #dA_e.append(ds_nm.coarsen(xh=args.nagg).sum().coarsen(yh=args.nagg).sum())
                dA_e.append(block_reshape_da_yx(ds_nm, nm, args.nagg, how='sum'))
            elif s_ == 'Cu':
                dA_e.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).sum())
                #dA_e.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='sum'))
            elif s_ == 'Cv':
                #dA_e.append(ds_nm[:,:,::args.nagg,:].coarsen(xh=args.nagg).sum())
                dA_e.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='sum'))
            elif s_ == 'q':
                print(nm,' at corners')
        elif cell_methods(ds_nm).find('mean')>-1:
            #dA_i_.append(ds_nm)
            if s_ == 'h':
                #dA_i.append(ds_nm.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean())
                #dA_i.append(ds_nm.coarsen(yh=args.nagg).mean().coarsen(xh=args.nagg).mean())
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='h', how='mean'))
                #dA_i.append(block_coarsen_xy(ds_nm, args.nagg, how='mean'))
                #dA_i.append(block_coarsen_yx(ds_nm, args.nagg, how='mean'))
                #dA_i.append(block_reshape_np_yx(ds_nm, nm, args.nagg, how='mean'))
                #dA_i.append(block_reshape_np_xy(ds_nm, nm, args.nagg, how='mean'))
                #dA_i.append(block_reshape_torch_yx(ds_nm, nm, args.nagg, how='mean'))
            elif s_ == 'Cu':
                #dA_i.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='mean'))
            elif s_ == 'Cv':
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='mean'))
            elif s_ == 'q':
                dA_i.append(ds_nm[:,:,::args.nagg,::args.nagg])
        else:
            print(nm,' has no cell method or an unrecognized cell method. Treating as mean.')
            #dA_i_.append(ds_nm)
            if s_ == 'h':
                #dA_i.append(ds_nm.coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean())
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='h', how='mean'))
            elif s_ == 'Cu':
                #dA_i.append(ds_nm[:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cu', how='mean'))
            elif s_ == 'Cv':
                #dA_i.append(ds_nm[:,:,::args.nagg,:].coarsen(xh=args.nagg).mean())
                dA_i.append(block_reshape_da_yx(ds_nm, nm, args.nagg, stagger='Cv', how='mean'))
            elif s_ == 'q':
                dA_i.append(ds_nm[:,:,::args.nagg,::args.nagg])

        #release the memory used by ds_nm
        del ds_nm

    print_mem('After computeing coarsened variables')
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

    average_T1=ds['average_T1']
    average_T2=ds['average_T2']
    average_DT=ds['average_DT']
    time_bnds=ds['time_bnds']


    ds_out=xr.merge([xr.merge(dA_e),xr.merge(dA_i),average_T1,average_T2,average_DT,time_bnds])

    nchunks=ntime
    chunk_size=int(ntime/nchunks)
    tstart=range(0,ntime,chunk_size)

    p=[]
    for n in np.arange(nchunks):
        p.append(multiprocessing.Process(target=save_timeslice,args=(ds_out,tstart[n],chunk_size,ntime)))
    print("Deubg: Here1")
    for p_ in p:
        p_.start()
    print("Deubg: Here2")
    for p_ in p:
        p_.join()

    print(' Multicore process complete.')


    rss = print_mem('At the end of the script')
    t2 = time.time()
    elapsed = int(t2 - t1)
    mem_gb = rss / 1024.0 / 1024.0 / 1024.0
    print(f"It took {elapsed} seconds to run on host {socket.gethostname()} using device {device}, consumed memory {mem_gb:5.2f} GB, using {n_workers * threads_per_worker} dask workers, checksum: {checksum}")

#command line example and timing result:
#/usr/bin/time -v python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/coarsen_ocean_z_month_simple.py -f ./01660101.ocean_z_month.nc --staticFile ./01660101.ocean_static.nc --nagg 3 --varname thetao --dask_workers 1
#
#It took 539 seconds to run on host an200 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5766339584.0
#It took 761 seconds to run on host an200 using device cpu, consumed memory  6.11 GB, using 2 dask workers, checksum: 5766339584.0 (system in use by others)
# 
#It took 311 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5766339584.0
#It took 263 seconds to run on host pp401 using device cpu, consumed memory  6.13 GB, using 2 dask workers, checksum: 5766339584.0
#3 and 4 workers crashed
#
#Using Copilot's block coarsen function:
#It took 205 seconds to run on host pp401 using device cpu, consumed memory 59.44 GB, using 1 dask workers, checksum: 5631767552.0
#Copi: The checksum is different because the block coarsen function does not trim the edges, so it includes some extra values in the mean that are not included in the xarray coarsen. This is expected and not a problem as long as we are consistent in how we compute the coarsened values.
#Niki: The above comment is fron Copi too :) Is they halucinating and try to explain the checksum difference? 
#Copi: I think the block coarsen function should trim the edges to be consistent with xarray coarsen
#Niki: 
#It took 235 seconds to run on host pp401 using device cpu, consumed memory  6.13 GB, using 2 dask workers, checksum: 5766339584.0
#
#subs
#orig  : It took 321 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#_xrT  : It took 266 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0 
#dask_xr 2 workers: It took 218 seconds to run on host pp401 using device cpu, consumed memory  6.13 GB, using 2 dask workers, checksum: 5766339584.0
# #_np   : It took 165 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5631767552.0
#_torch: It took 153 seconds to run on host pp401 using device cpu, consumed memory 59.66 GB, using 1 dask workers, checksum: 5631767552.0
#
#02122026
#xr.coarsen_yx :It took 333 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#suspect xr.coarsen_xy :It took 333 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#block_reshape_xy : It took 331 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#block_reshape_yx : It took 275 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#block_reshape_np : It took 218 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5631767552.0
#
#02172026 year 198
# /usr/bin/time -v python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/coarsen_ocean_z_month_simple.py -f ./01980101.ocean_z_month.nc --staticFile ./01980101.ocean_static.nc --nagg 3 --varname thetao
#xr.coarsen_xy : It took 341 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720640.0
#xr.coarsen_yx : It took 279 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720128.0
#block_coarsen_xy : It took 324 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720640.0
#block_coarsen_yx : It took 255 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720128.0
#block_reshape_np_xy : It took 221 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_np_yx : It took 170 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_np_yx : It took 204 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_np_yx  6vars: It took 1607 seconds to run on host pp401 using device cpu, consumed memory 246.97 GB, using 1 dask workers, checksum: 2710650350911424.0
#block_reshape_torch_yx: It took 149 seconds to run on host pp401 using device cpu, consumed memory 59.66 GB, using 1 dask workers, checksum: 5668890624.0
#numpy+dask,  numpy answers, less memory foot print even with 1 worker!
#block_reshape_da_yx : It took 135 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_da_yx 1var : It took 147 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_da_yx 2vars: It took 276 seconds to run on host pp401 using device cpu, consumed memory 12.22 GB, using 1 dask workers, checksum: 30006976512.0
#block_reshape_da_yx 3vars: It took 474 seconds to run on host pp401 using device cpu, consumed memory 18.16 GB, using 1 dask workers, checksum: 67230810112.0
#block_reshape_da_yx 6vars: It took 1400 seconds to run on host pp401 using device cpu, consumed memory 36.20 GB, using 1 dask workers, checksum: 2710650350911424.0
#block_reshape_da_yx 6vars_CuCv: It took 1513 seconds to run on host pp401 using device cpu, consumed memory 36.12 GB, using 1 dask workers, checksum: 2710719237373888.0
#block_reshape_np_yx 6vars: It took 1607 seconds to run on host pp401 using device cpu, consumed memory 246.97 GB, using 1 dask workers, checksum: 2710650350911424.0
#
#thetao
#(plattorch) Niki.Zadeh: /xtmp/Niki.Zadeh/work/William.Cooke/SPEAR/SPEAR_HI_8/SPEAR_c384_OM4p08_Control_1990_A14 $  /usr/bin/time -v python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/coarsen_ocean_z_month_simple.py -f ./01980101.ocean_z_month.nc --staticFile ./01980101.ocean_static.nc --nagg 3 --varname thetao
#coarsen_original         : It took 354 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5804720640.0
#coarsen_original_yx.compute dask_workers 0: 
#coarsen_original_yx.compute dask_workers 1: Hangs at writing outputs on PAN
#          : 
#block_reshape_da_yx 1var : It took 170 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 5668890624.0
#umo
#coarsen_original         : It took 429 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 2711735513382912.0
#block_reshape_da_yx 1var : It took 412 seconds to run on host pp401 using device cpu, consumed memory  6.50 GB, using 1 dask workers, checksum: 2713882191724544.0
#block_reshape_da_yx_TR 1var : It took 421 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 2711735513382912.0
#
#02182026 year 198, using host c5n1842
#block_reshape_np_yx : It took 169 seconds to run on host c5n1842 using device cpu, consumed memory 59.35 GB, using 1 dask workers, checksum: 5668890624.0
#block_coarsen_yx :    It took 325 seconds to run on host c5n1842 using device cpu, consumed memory 59.36 GB, using 1 dask workers, checksum: 5804720128.0
#block_coarsen_yx :    It took 404 seconds to run on host c5n1842 using device cpu, consumed memory  6.66 GB, using 2 dask workers, checksum: 5804720128.0
#block_coarsen_yx :    It took 242 seconds to run on host c5n1842 using device cpu, consumed memory  6.70 GB, using 4 dask workers, checksum: 5804720128.0
#block_coarsen_yx :    It took 187 seconds to run on host c5n0793 using device cpu, consumed memory  6.73 GB, using 8 dask workers, checksum: 5804720128.0
#block_coarsen_yx :    It took 138 seconds to run on host c5n0793 using device cpu, consumed memory  6.75 GB, using 12 dask workers, checksum: 5804720128.0
#block_coarsen_yx :    
#block_reshape_da_yx : It took 217 seconds to run on host c5n0793 using device cpu, consumed memory  6.73 GB, using 8 dask workers, checksum: 5668890624.0
#block_reshape_da_yx : It took 177 seconds to run on host c5n0793 using device cpu, consumed memory  6.75 GB, using 12 dask workers, checksum: 5668890624.0

#block_reshape_da_yx : It took 204 seconds to run on host c5n0793 using device cpu, consumed memory  6.35 GB, using 1 dask workers, checksum: 5668890624.0
#block_reshape_da_yx : It took 179 seconds to run on host c5n0793 using device cpu, consumed memory  6.76 GB, using 12 dask workers, checksum: 5668890624.0
#
#block_reshape_np_yx 6vars: It took 2383 seconds to run on host c5n0793 using device cpu, consumed memory 36.24 GB, using 1 dask workers, checksum: 2710650350911424.0
#block_reshape_np_yx 6vars: It took 1086 seconds to run on host c5n0793 using device cpu, consumed memory 36.43 GB, using 12 dask workers, checksum: 2710719237373888.0
#
#02202026
#thetao
#(plattorch) Niki.Zadeh: /xtmp/Niki.Zadeh/work/William.Cooke/SPEAR/SPEAR_HI_8/SPEAR_c384_OM4p08_Control_1990_A14 $  /usr/bin/time -v python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/coarsen_ocean_z_month_simple.py -f ./01980101.ocean_z_month.nc --staticFile ./01980101.ocean_static.nc --nagg 3 --varname thetao
#coarser_original_xy : 
#  Compute took 350 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 0 dask workers, checksum: 5804720640.0
#  It took 361 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 0 dask workers, checksum: 5804720640.0
#                      top shows memory usage up to 170GB, but the final RSS is 59.38 GB, which suggests that there may be some temporary memory spikes during the coarsening process
#coarser_original_yx : 
#  Compute took 314 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 0 dask workers, checksum: 5804720128.0
#  It took 323 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 0 dask workers, checksum: 5804720128.0
#    top shows memory usage up to 170GB
#coarser_original_yx.compute : 
#  Compute took 268 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 0 dask workers, checksum: 5804720128.0
#  It took 276 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 0 dask workers, checksum: 5804720128.0
#coarser_original_yx.compute.time_chunk_size_1 : Hangs at writing outputs on PAN
#  Compute took 172 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 5804720128.0
#  Hangs at writing outputs on PAN
#      top shows memory usage up to just under 20GB
#      ds=xr.open_dataset(path, decode_times = False, chunks={'time': 1}) activates dask chunking along the time dimension, which can help reduce memory usage by processing one time step at a time. This is likely why the memory usage is much lower in this case compared to when no chunking is used.                       
#block_reshape_da_yx :
#  Compute took 163 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 0 dask workers, checksum: 5668890624.0
#  It took 173 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 0 dask workers, checksum: 5668890624.0
#block_reshape_da_yx --dask_workers 1 :
#  Compute took 158 seconds to run on host pp401 using device cpu, consumed memory  6.30 GB, using 1 dask workers, checksum: 5668890624.0
#  Hangs at writing outputs on PAN