import xarray as xr
import glob
import numpy as np
import geopandas as gpd
import pandas as pd
import sys
sys.path.append('../')
from utilities.era5_down import nc_inter_2D, cropped_nc, add_variable_along_timelatlon, annual_smb_glacier
from utilities.plot_results import plot_smb


dso2 = xr.open_dataset('../data/ERA_59_20_day_int/Belling_1959_2020.nc')
df_g = gpd.read_file('../data/static/Shapefiles/SSI_all_fff_20221118.shp')
df_smb = pd.read_csv('../data/mass_balance/SSI_SMB.csv', sep='\t', index_col=['YEAR'])

glacier = df_g[df_g.Name == 'Belling Glacier']

ds = cropped_nc(dso2, glacier)

ds.to_netcdf('../data/ERA_59_20_day_int/Belling_1959_2020_sel.nc')

glaciers_shp = ['Hurd Glacier', 'Johnsons Glacier', 'Belling Glacier']
glaciers_smb = ['HURD', 'JOHNSONS', 'BELLINGSHAUSEN']
df_all = annual_smb_glacier(df_g, glaciers_shp[2], dso2, df_smb, glaciers_smb[2])
fig = plot_smb(df_all)
# Finalmente guardamos nuestra figura
print('Done:'+' '+ glaciers_smb[2])
fig.savefig('../fig/best'+'_'+glaciers_smb[2] +'.png', dpi = 200, facecolor='w', bbox_inches = 'tight', 
                pad_inches = 0.1)
