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

factor_melt = -0.01415625,

dso2  = dso.copy()
dso2  = dso2.where(dso2['MASK']==1, np.nan) #[['T2','PDD','SF']]
SMB   = dso2['SF'] + (dso2['PDD'] * factor_melt)
MO    = ((0.003 * dso2['T2'] + 0.52)* dso2['SF'])
MELT  = (dso2['PDD'] * factor_melt)
RF    = MELT + MO

add_variable_along_timelatlon(dso2, MELT.values, 'MELT', 'm of water equivalent', 'Melt')
add_variable_along_timelatlon(dso2, SMB.values, 'SMB', 'm of water equivalent', 'Surface Mass Banlance')
add_variable_along_timelatlon(dso2, RF.values, 'Q', 'm of water equivalent', 'Runoff')
add_variable_along_timelatlon(dso2, MO.values, 'RZ', 'm of water equivalent', 'Refreezing')

dso2.to_netcdf('../data/ERA_59_20_day_int/Belling_1959_2020.nc')

breakpoint()

df_g = gpd.read_file('../data/static/Shapefiles/SSI_all_fff_20221118.shp')
df_smb = pd.read_csv('../data/mass_balance/SSI_SMB.csv', sep='\t', index_col=['YEAR'])

glaciers_shp = ['Hurd Glacier', 'Johnsons Glacier', 'Belling Glacier']
glaciers_smb = ['HURD', 'JOHNSONS', 'BELLINGSHAUSEN']
df_all = annual_smb_glacier(df_g, glaciers_shp[2], dso2, df_smb, glaciers_smb[2])
fig = plot_smb(df_all)
# Finalmente guardamos nuestra figura
print('Done:'+' '+ glaciers_smb[2])
fig.savefig('../fig/'+str(jj)+'_'+glaciers_smb[2] +'.png', dpi = 200, facecolor='w', bbox_inches = 'tight', 
                pad_inches = 0.1)

breakpoint()

factor_melt = np.linspace(-0.0145, -0.013125, 5)
print(factor_melt)
for jj in range(len(factor_melt)):

    dso2  = dso.copy()
    dso2  = dso2.where(dso2['MASK']==1, np.nan) #[['T2','PDD','SF']]
    SMB   = dso2['SF'] + (dso2['PDD'] * factor_melt[jj])
    MO    = ((0.003 * dso2['T2'] + 0.52)* dso2['SF'])
    MELT  = (dso2['PDD'] * factor_melt[jj])
    RF    = MELT + MO

    add_variable_along_timelatlon(dso2, MELT.values, 'MELT', 'm of water equivalent', 'Melt')
    add_variable_along_timelatlon(dso2, SMB.values, 'SMB', 'm of water equivalent', 'Surface Mass Banlance')
    add_variable_along_timelatlon(dso2, RF.values, 'Q', 'm of water equivalent', 'Runoff')
    add_variable_along_timelatlon(dso2, MO.values, 'RZ', 'm of water equivalent', 'Refreezing')

    #dso.to_netcdf('../data/ERA_59_20_day_int/int_era5_1959_2020.nc')
    #breakpoint()

    df_g = gpd.read_file('../data/static/Shapefiles/SSI_all_fff_20221118.shp')
    df_smb = pd.read_csv('../data/mass_balance/SSI_SMB.csv', sep='\t', index_col=['YEAR'])

    glaciers_shp = ['Hurd Glacier', 'Johnsons Glacier', 'Belling Glacier']
    glaciers_smb = ['HURD', 'JOHNSONS', 'BELLINGSHAUSEN']
    df_all = annual_smb_glacier(df_g, glaciers_shp[2], dso2, df_smb, glaciers_smb[2])
    fig = plot_smb(df_all)
    # Finalmente guardamos nuestra figura
    print('Done:'+' '+ glaciers_smb[2])
    fig.savefig('../fig/'+str(jj)+'_'+glaciers_smb[2] +'.png', dpi = 200, facecolor='w', bbox_inches = 'tight', 
                    pad_inches = 0.1)

#    for i in range(len(glaciers_shp)):
#        df_all = annual_smb_glacier(df_g, glaciers_shp[i], dso2, df_smb, glaciers_smb[i])
#        fig = plot_smb(df_all)
#        # Finalmente guardamos nuestra figura
#        print('Done:'+' '+ glaciers_smb[i])
#        fig.savefig('../fig/'+str(jj)+'_'+glaciers_smb[i] +'.png', dpi = 200, facecolor='w', bbox_inches = 'tight', 
#                    pad_inches = 0.1)
