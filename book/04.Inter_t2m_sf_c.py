import xarray as xr
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
import sys
sys.path.append('../')
from utilities.era5_down import nc_inter_2D, add_variable_along_timelatlon, annual_smb_glacier
from utilities.plot_results import plot_smb

input_dem = '../data/static/SSI_static_1200_500.nc'
ds_dem = xr.open_dataset(input_dem)

files = sorted(glob.glob('../data/ERA_59_20_day/*.nc'))

ds = xr.open_dataset(files[0])
dso = nc_inter_2D(ds, ds_dem, 'nearest')
print('Done:'+' '+ files[0][22:])

for i in range(len(files))[1:]:
    ds = xr.open_dataset(files[i])
    dso1 = nc_inter_2D(ds, ds_dem, 'nearest')
    dso = xr.merge([dso, dso1])
    #print(dso)
    print('Done:'+' '+ files[i][22:])
dso.to_netcdf('../data/ERA_59_20_day_int/t2m_sf_1959_2020.nc')
