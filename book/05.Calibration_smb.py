import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
import sys
sys.path.append('../')
from utilities.era5_down import nc_inter_2D, cropped_nc, add_variable_along_timelatlon, annual_smb_glacier
from utilities.plot_results import plot_smb

#factor_melt = np.linspace(-0.0145, -0.013125, 5)
#print(factor_melt)
df_g = gpd.read_file('../data/static/Shapefiles/SSI_all_fff_20221118.shp')
df_smb = pd.read_csv('../data/mass_balance/SSI_SMB.csv', sep='\t', index_col=['YEAR'])
glaciers_shp = ['Hurd Glacier', 'Johnsons Glacier', 'Belling Glacier']
glaciers_smb = ['HURD', 'JOHNSONS', 'BELLINGSHAUSEN']
dso = xr.open_dataset('../data/ERA_59_20_day_int/t2m_sf_1959_2020.nc')

for g in range(len(glaciers_shp)):

    glacier = df_g[df_g.Name == glaciers_shp[g]]
    dsog = cropped_nc(dso, glacier) 

    a = -0.05
    b = -0.001
    n   = 100

    factor_melt = np.linspace(a, b, n)
    r2_list = []
    rmse_list = []
    for i in range(n):

        dsog1   = dsog.where(dsog['MASK']==1, np.nan)
        SMB    = dsog1['SF'] + (dsog1['PDD'] * factor_melt[i])
        add_variable_along_timelatlon(dsog1, SMB.values, 'SMB', 'm of water equivalent', 'Surface Mass Banlance')
        df_all = annual_smb_glacier(dsog1, df_smb, glaciers_smb[g])
        df_all1 = df_all.copy().dropna()
        r2_list.append(r2_score(df_all1['OBS'].values, df_all1['SIM'].values))
        rmse_list.append(mean_squared_error(df_all1['OBS'].values, df_all1['SIM'].values))

    df_cal = pd.DataFrame({'FM': factor_melt, 'r2': r2_list, 'RMSE': rmse_list})
    df_cal.to_csv('../data/out/' + glaciers_smb[g] + 'cal.csv', sep='\t', index=False)
    print(glaciers_smb[g])
    print(df_cal.loc[df_cal['r2'].sub(1).abs().idxmin()])

    best_FM = df_cal.loc[df_cal['r2'].sub(1).abs().idxmin()]['FM']

    dsog1 = dsog.where(dsog['MASK']==1, np.nan)
    SMB   = dsog1['SF'] + (dsog1['PDD'] * best_FM)
    MO    = ((0.003 * dsog1['T2'] + 0.52)* dsog1['SF'])
    MELT  = (dsog1['PDD'] * best_FM)
    RF    = MELT + MO
    RF    = RF.where(RF < 0, 0).where(dsog['MASK']==1, np.nan)

    add_variable_along_timelatlon(dsog1, SMB.values, 'SMB', 'm of water equivalent', 'Surface Mass Banlance')
    add_variable_along_timelatlon(dsog1, -MELT.values, 'MELT', 'm of water equivalent', 'Melt')
    add_variable_along_timelatlon(dsog1, -RF.values, 'Q', 'm of water equivalent', 'Runoff')
    add_variable_along_timelatlon(dsog1, MO.values, 'RZ', 'm of water equivalent', 'Refreezing')

    #dso.to_netcdf('../data/ERA_59_20_day_int/int_era5_1959_2020.nc')
    dsog1.to_netcdf('../data/out/' + glaciers_smb[g] + '_1960_2020.nc')
