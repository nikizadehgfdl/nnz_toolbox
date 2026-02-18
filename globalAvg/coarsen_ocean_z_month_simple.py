##
##Script to coarsen 2D data. Based on a tool from Will Cooke and Matt Harison used for 1/12 degree ocean model output..
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

def block_coarsen_np_yx(da_xr, var_name, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using numpy reshape.
    """
    print(f"block_coarsen_np: how={how}, nagg={nagg}")
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

def block_coarsen_np_xy(da_xr, var_name, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using numpy reshape.
    """
    print(f"block_coarsen_np: how={how}, nagg={nagg}")
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

def block_coarsen_torch_yx(da_xr, var_name, nagg, how='mean',device='cpu'):
    """Fast block coarsen for 'h' staggered fields using torch tensor reshape.
    """
    import torch
    device = torch.device(device) 

    print(f"block_coarsen_torch: how={how}, nagg={nagg}")
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

        
#Provided by Copilot
def block_coarsen_da(da_xr, nagg, how='mean'):
    """Fast block coarsen for 'h' staggered fields using numpy reshape.

    - Trims edges if dimensions are not divisible by `nagg`.
    - Falls back to xarray.coarsen if the backing array is a dask array.
    """
    try:
        import dask.array as _dask_array
        is_dask = hasattr(da_xr.data, 'chunks')
    except Exception:
        is_dask = False
    print(f"block_coarsen_da: is_dask={is_dask}, how={how}, nagg={nagg}")

    if is_dask:
        if how == 'sum':
            return da_xr.coarsen(xh=nagg).sum().coarsen(yh=nagg).sum().compute()
        else:
            return da_xr.coarsen(xh=nagg).mean().coarsen(yh=nagg).mean().compute()

    arr = da_xr.values
    if arr is None:
        return da_xr

    dims = list(da_xr.dims)
    if 'xh' not in dims or 'yh' not in dims:
        return da_xr

    ix = dims.index('xh')
    iy = dims.index('yh')

    # move yh,xh to last two axes
    arr_moved = np.moveaxis(arr, (iy, ix), (-2, -1))
    ny, nx = arr_moved.shape[-2], arr_moved.shape[-1]
    ny_trim = (ny // nagg) * nagg
    nx_trim = (nx // nagg) * nagg
    if ny_trim == 0 or nx_trim == 0:
        return da_xr

    arr_trim = arr_moved[..., :ny_trim, :nx_trim]
    newshape = arr_trim.shape[:-2] + (ny_trim // nagg, nagg, nx_trim // nagg, nagg)
    arr_reshaped = arr_trim.reshape(newshape)
    if how == 'sum':
        out = arr_reshaped.sum(axis=(-1, -3))
    else:
        out = arr_reshaped.mean(axis=(-1, -3))

    # move reduced axes back to original positions
    out = np.moveaxis(out, (-2, -1), (iy, ix))

    # build new coords: trim original coords and reduce
    coords = {}
    for k in da_xr.coords:
        try:
            coords[k] = da_xr.coords[k]
        except Exception:
            pass

    # adjust xh and yh coords to block centers (mean of each block)
    xh_vals = da_xr['xh'].values[:nx_trim].reshape(nx_trim // nagg, nagg).mean(axis=1)
    yh_vals = da_xr['yh'].values[:ny_trim].reshape(ny_trim // nagg, nagg).mean(axis=1)
    coords['xh'] = xh_vals
    coords['yh'] = yh_vals

    new_dims = list(da_xr.dims)
    # replace lengths implicitly by returning DataArray with new coord sizes
    return xr.DataArray(out, dims=new_dims, coords={d: (coords[d] if d in coords else da_xr.coords[d]) for d in new_dims}, attrs=da_xr.attrs, name=da_xr.name)

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

def calc_w_from_convergence(u_var, v_var, wrapx = True, wrapy = False):

  tmax = u_var.shape[0]

  ushape = u_var.shape
  vshape = v_var.shape
  is_symmetric = True
  if ushape[3] == vshape[3]:
    is_symmetric = False

  ntime, nk, nlat, nlon = u_var.shape
  if is_symmetric:
    w = np.ma.zeros( (ntime, nk+1, nlat, nlon-1)  )
  else:
    w = np.ma.zeros( (ntime, nk+1, nlat, nlon)  )
  # Work timelevel by timelevel
  for tidx in range(0,tmax):
    # Get and process the u component
    if is_symmetric:
      u_dat = u_var[tidx,:,:,1:]
    else:
      u_dat = u_var[tidx,:,:,:]
    #h_mask = np.logical_or(np.ma.getmask(u_dat), np.ma.getmask(np.roll(u_dat,1,axis=-1)))
    u_dat = u_dat.filled(0.)

    # Get and process the v component
    if is_symmetric:
      v_dat = v_var[tidx,:,1:,:]
    else:
      v_dat = v_var[tidx,:,:,:]
    #h_mask = np.logical_or(h_mask,np.ma.getmask(v_dat))
    #h_mask = np.logical_or(h_mask,np.ma.getmask(np.roll(v_dat,1,axis=-2)))
    v_dat = v_dat.filled(0.)

    # Order of subtraction based on upwind sign convention and desire for w>0 to correspond with upwards velocity
    w[tidx,:-1,:,:] += np.roll(u_dat,1,axis=-1)-u_dat
    if not wrapx: # If not wrapping, then convergence on westernmost side is simply so subtract back the rolled value
      w[tidx,:-1,:,0] += -u_dat[:-1,:,-1]
    w[tidx,:-1,:,:] += np.roll(v_dat,1,axis=-2)-v_dat
    if not wrapy: # If not wrapping, convergence on westernmost side is v
      w[tidx,:-1,0,:] += -v_dat[:,-1,:]
    w[tidx,-1,:,:] = 0.
    # Do a double-flip so that we integrate from the bottom
    w[tidx,:-1,:,:] = w[tidx,-2::-1,:,:].cumsum(axis=0)[::-1,:,:]
    # Mask if any of u[i-1], u[i], v[j-1], v[j] are not masked
    #w[tidx,:-1,:,:] = np.ma.masked_where(h_mask, w[tidx,:-1,:,:])
    # Bottom should always be zero, mask applied wherever the top interface is a valid value
    #w[tidx,-1,:,:] = np.ma.masked_where(h_mask[-2,:,:], w[tidx,-1,:,:])

  return w

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Script for implied atmospheric meridional moisture transport.''')
    parser.add_argument('-f','--filename', type=str, default='foo.nc', help='''filename ''')
    parser.add_argument('--staticFile', type=str, default='ocean_static_aggx3.nc', help='''filename ''')
    parser.add_argument('-n','--nagg', type=int, default=3, help='''year ''')
    parser.add_argument('--varname', type=str, default='all', help='''Name of the variable to coarsen. Default is "all", which means all variables will be coarsened.''')
    parser.add_argument('--uname', type=str, default='umo', help='''Name of the u-component of mass transport''')
    parser.add_argument('--vname', type=str, default='vmo', help='''Name of the v-component of mass transport''')
    parser.add_argument('--wrapx', type=bool, default=True, help='''True if the x-component is reentrant''')
    parser.add_argument('--wrapy', type=bool, default=False, help='''True if the y-component is reentrant''')
    parser.add_argument('--dask_workers', type=int, default=1, help='''Number of Dask workers to use. Default is 1, which means no Dask parallelization.''')

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
    if args.dask_workers > 1:
        time_chunk_size = 1  # Adjust this based on your memory constraints and dataset size
    ds=xr.open_dataset(path, decode_times = False, chunks={'time': time_chunk_size} if args.dask_workers > 1 else None)
    static_path=args.staticFile
    ds_static=xr.open_dataset(static_path)
    dA_i=[];dA_e=[];dA_i_=[];dA_e_=[]

    print_mem('After open_dataset calls')
    exclude_list=['uo','vo','wo'] # These will be calculated later.

    ntime=ds['thetao'].shape[0]
    for nm in ds.data_vars:
        if nm in exclude_list:
            print(nm+" will be calculated later. Ignoring variable in original file.")
            continue
        if args.varname != 'all' and nm != args.varname:
            print(nm+" was not requested in the args. Ignoring variable in original file.")
            continue
        s_=stagger(ds[nm])
        print(nm,s_)

        if cell_methods(ds[nm]).find('sum')>-1:
            dA_e_.append(ds[nm])
            if s_ == 'h':
                dA_e.append(ds[nm].coarsen(xh=args.nagg).sum().coarsen(yh=args.nagg).sum())
            elif s_ == 'Cu':
                dA_e.append(ds[nm][:,:,:,::args.nagg].coarsen(yh=args.nagg).sum())
            elif s_ == 'Cv':
                dA_e.append(ds[nm][:,:,::args.nagg,:].coarsen(xh=args.nagg).sum())
            elif s_ == 'q':
                print(nm,' at corners')
        elif cell_methods(ds[nm]).find('mean')>-1:
            dA_i_.append(ds[nm])
            if s_ == 'h':
                #dA_i.append(ds[nm].coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean())
                #dA_i.append(ds[nm].coarsen(yh=args.nagg).mean().coarsen(xh=args.nagg).mean())
                #dA_i.append(block_coarsen_xy(ds[nm], args.nagg, how='mean'))
                #dA_i.append(block_coarsen_yx(ds[nm], args.nagg, how='mean'))
                #dA_i.append(block_coarsen_np_yx(ds[nm], nm, args.nagg, how='mean'))
                #dA_i.append(block_coarsen_np_xy(ds[nm], nm, args.nagg, how='mean'))
                dA_i.append(block_coarsen_torch_yx(ds[nm], nm, args.nagg, how='mean'))
            elif s_ == 'Cu':
                dA_i.append(ds[nm][:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
            elif s_ == 'Cv':
                dA_i.append(ds[nm][:,:,::args.nagg,:].coarsen(xh=args.nagg).mean())
            elif s_ == 'q':
                dA_i.append(ds[nm][:,:,::args.nagg,::args.nagg])
        else:
            dA_i_.append(ds[nm])
            if s_ == 'h':
                dA_i.append(ds[nm].coarsen(xh=args.nagg).mean().coarsen(yh=args.nagg).mean())
            elif s_ == 'Cu':
                dA_i.append(ds[nm][:,:,:,::args.nagg].coarsen(yh=args.nagg).mean())
            elif s_ == 'Cv':
                dA_i.append(ds[nm][:,:,::args.nagg,:].coarsen(xh=args.nagg).mean())
            elif s_ == 'q':
                dA_i.append(ds[nm][:,:,::args.nagg,::args.nagg])

    print_mem('After computeing coarsened variables')

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

    # make a checksum to compare with different runs (safe for dask-backed arrays)
    checksum = sum([sum_to_scalar(d) for d in dA_i]) + sum([sum_to_scalar(d) for d in dA_e])

    nchunks=ntime
    chunk_size=int(ntime/nchunks)
    tstart=range(0,ntime,chunk_size)

    p=[]
    for n in np.arange(nchunks):
        p.append(multiprocessing.Process(target=save_timeslice,args=(ds_out,tstart[n],chunk_size,ntime)))

    for p_ in p:
        p_.start()

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
#block_coarsen_xy : It took 331 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#block_coarsen_yx : It took 275 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5766339584.0
#block_coarsen_np : It took 218 seconds to run on host pp401 using device cpu, consumed memory 59.37 GB, using 1 dask workers, checksum: 5631767552.0
#
#02172026 year 198
# /usr/bin/time -v python /nbhome/Niki.Zadeh/projects/nnz_toolbox/globalAvg/coarsen_ocean_z_month_simple.py -f ./01980101.ocean_z_month.nc --staticFile ./01980101.ocean_static.nc --nagg 3 --varname thetao
#xr.coarsen_xy : It took 341 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720640.0
#xr.coarsen_yx : It took 279 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720128.0
#block_coarsen_xy : It took 324 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720640.0
#block_coarsen_yx : It took 255 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5804720128.0
#block_coarsen_np_yx : It took 170 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5668890624.0
#block_coarsen_np_xy : It took 221 seconds to run on host pp401 using device cpu, consumed memory 59.38 GB, using 1 dask workers, checksum: 5668890624.0
#block_coarsen_torch_yx: It took 149 seconds to run on host pp401 using device cpu, consumed memory 59.66 GB, using 1 dask workers, checksum: 5668890624.0