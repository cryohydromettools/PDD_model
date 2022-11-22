import xarray as xr
import pandas as pd
from scipy.stats import pearsonr
import numpy as np

def era5_down(files,lon, lat, elev):
    ds = xr.open_mfdataset(files)
    df = ds.sel(longitude=lon, 
                latitude=lat, 
                method='nearest').to_dataframe()
    g       = 9.80665
    hgt_era = df['z'].values[0]/g
    hgt_aws = elev
    df['t2m'] = (df['t2m'].values + (hgt_aws - hgt_era) * -0.009) - 273.16

    # snowfall
    snowfall = df['sf'].values #+ (hgt_aws - hgt_era) * 0.000005
    snowfall[snowfall < 0]  = 0.0
    df['sf'] = snowfall
    
    df = df[['t2m','sf']]
    
    return df

def cropped_nc(ds, glacier):
    bound = glacier.total_bounds
    min_lon = bound[0]
    min_lat = bound[1]
    max_lon = bound[2]
    max_lat = bound[3]
    mask_lon = (ds.lon >= min_lon) & (ds.lon <= max_lon)
    mask_lat = (ds.lat >= min_lat) & (ds.lat <= max_lat)
    cropped_hurd = ds.where(mask_lon & mask_lat, drop=True)
    return cropped_hurd

def sel_glacier(df1, name):
    subset = df1.loc[lambda df1: df1['NAME'] == name][['WINTER_BALANCE','SUMMER_BALANCE', 'NAME']]
    years = []
    names = []
    for i in subset.index:
        for j in range(2):
            years.append(i)
            names.append(subset['NAME'].values[0])
    df_data = pd.DataFrame(subset[['WINTER_BALANCE','SUMMER_BALANCE']].values.flatten(),
                           columns={'SMB'}, index = years)
    df_data['Glacier'] = names
    df_data.index.name = 'Year'
    
    return df_data

def add_variable_along_timelatlon(ds, var, name, units, long_name):
    ds[name] = (('time','lat','lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

    return ds

def nc_inter_2D(ds_int, ds_dem, method):

    ds_int = ds_int[['z', 't2m', 'sf']].interp(longitude=ds_dem.lon, latitude=ds_dem.lat,
    method=method, kwargs={'fill_value': 'extrapolate'})

    dem_era5 = ds_int['z'][0].values/9.80665
    dem_era5[dem_era5 < 0] = 0

    t2m_int = (ds_int['t2m'] + (ds_dem['HGT'].values - dem_era5) * - 0.009) - 273.16
    ppd_int = t2m_int.where(t2m_int >= 0, 0)    
    ppd_int = ppd_int.resample(time='1MS').sum()
    t2m_int = t2m_int.resample(time='1MS').mean()

    sf_int = ds_int['sf'] #+ (ds_dem['HGT'].values - dem_era5) * 0.000001
    sf_int = sf_int.where(sf_int >= 0, 0)
    sf_int = sf_int.resample(time='1MS').sum()
    #sf_int  = ds_int['sf'].resample(time='1MS').sum()

    dso = ds_dem.copy()
    dso.coords['time'] = t2m_int['time'].values

    add_variable_along_timelatlon(dso, t2m_int.values, 'T2', 'K', 'Temperature at 2 m')
    add_variable_along_timelatlon(dso, ppd_int.values, 'PDD', 'D', 'Days')
    add_variable_along_timelatlon(dso, sf_int.values, 'SF', 'm of water equivalent', 'Snowfall')

    return dso

def annual_smb_glacier(dso, df_smb, glacier_smb):
    df_sel = dso['SMB'].where(dso['MASK']==1).mean(('lon', 'lat')).to_dataframe()
    df_sel['Year'] = df_sel.index.year
    Year = df_sel.drop_duplicates(subset=['Year'])['Year'].values
    annual_smb = []
    for i in range(len(Year)-1):
        annual_smb.append(df_sel.loc[str(Year[i])+'0401':str(Year[i+1])+'0331']['SMB'].sum())
    df_data = pd.DataFrame(annual_smb, columns={'SIM'}, index = Year[1:])
    df_all = pd.merge(df_data, df_smb[glacier_smb], how='left', left_index=True, right_index=True)
    df_all.rename(columns={glacier_smb: 'OBS'}, inplace=True)

    return df_all

def annual_sum(df):
    df['Year'] = df.index.year
    Year = df.drop_duplicates(subset=['Year'])['Year'].values
    df.drop(['Year'], axis=1, inplace = True)
    var_sum = []
    for i in range(len(Year)-1):
        var_sum.append(df.loc[str(Year[i])+'0401':str(Year[i+1])+'0331'].sum())
    df_data = pd.DataFrame(var_sum, columns=df.columns, index = Year[1:])
    df_data.index.name = 'Year'

    return df_data

def annual_mean(df):
    df['Year'] = df.index.year
    Year = df.drop_duplicates(subset=['Year'])['Year'].values
    df.drop(['Year'], axis=1, inplace = True)
    var_sum = []
    for i in range(len(Year)-1):
        var_sum.append(df.loc[str(Year[i])+'0401':str(Year[i+1])+'0331'].mean())
    df_data = pd.DataFrame(var_sum, columns=df.columns, index = Year[1:])
    df_data.index.name = 'Year'

    return df_data

def annual_smb_glacier1(df_g, glacier_name, ds, df_smb, glacier_smb):
    glacier = df_g[df_g.Name == glacier_name]
    ds_sel = cropped_nc(ds, glacier)
    df_sel = ds_sel.mean(('lon', 'lat')).to_dataframe()
#    df_sel = ds_sel['SMB'].where(ds_sel['MASK']==1).mean(('lon', 'lat')).to_dataframe()
    df_sel['Year'] = df_sel.index.year
    Year = df_sel.drop_duplicates(subset=['Year'])['Year'].values
    annual_smb = []
    for i in range(len(Year)-1):
        annual_smb.append(df_sel.loc[str(Year[i])+'0401':str(Year[i+1])+'0331']['SMB'].sum())
    df_data = pd.DataFrame(annual_smb, columns={'SIM'}, index = Year[1:])
    df_all = pd.merge(df_data, df_smb[glacier_smb], how='left', left_index=True, right_index=True)
    df_all.rename(columns={glacier_smb: 'OBS'}, inplace=True)

    return df_all

def season_df(df_all, Season, start_data, end_data):
    seasons = {
        1: 'DJF', 2: 'DJF',
        3: 'MAM', 4: 'MAM', 5: 'MAM',
        6: 'JJA', 7: 'JJA', 8: 'JJA',
        9: 'SON', 10: 'SON', 11: 'SON',
        12: 'DJF'
    }
    df_all['Month'] = df_all.index.month
    df_all['Season'] = df_all['Month'].apply(lambda x: seasons[x])
    df_all['Year'] = df_all.index.year
    df_all['Year'] = df_all.apply(lambda x: (x['Year']+1) if x['Month'] == 12 else x['Year'], axis=1)
    df_all_sea = df_all.groupby(['Year', 'Season']).mean().reset_index()
    df_all_sea.index = df_all_sea['Season']
    df_all_sea = df_all_sea.loc[Season].reset_index(drop=True)
    df_all_sea.index = df_all_sea['Year']
    df_all_sea.drop(['Year', 'Season', 'Month'], axis = 1, inplace=True)
    df_all_sea = df_all_sea.loc[start_data:end_data]


    return df_all_sea

def sel_period(df_all, Period, start_data, end_data):
    seasons = {
        1: 'summer', 2: 'summer',
        3: 'summer', 4: 'summer', 5: 'winter',
        6: 'winter', 7: 'winter', 8: 'winter',
        9: 'winter', 10: 'winter', 11: 'winter',
        12: 'summer'
    }
    df_all['Month'] = df_all.index.month
    df_all['Period'] = df_all['Month'].apply(lambda x: seasons[x])
    df_all['Year'] = df_all.index.year
    df_all['Year'] = df_all.apply(lambda x: (x['Year']+1) if x['Month'] == 12 else x['Year'], axis=1)
    df_all_sea = df_all.groupby(['Year', 'Period']).mean().reset_index()
    df_all_sea.index = df_all_sea['Period']
    df_all_sea = df_all_sea.loc[Period].reset_index(drop=True)
    df_all_sea.index = df_all_sea['Year']
    df_all_sea.drop(['Year', 'Period', 'Month'], axis = 1, inplace=True)
    df_all_sea = df_all_sea.loc[start_data:end_data]

    return df_all_sea

def df_corr_pvalues(df_all):
    rho = df_all.corr()
    pval = df_all.corr(method=lambda x, y: pearsonr(x, y)[1]) - np.eye(*rho.shape)
    p = pval.applymap(lambda x: ''.join(['*' for t in [.05, .01, .001] if x<=t]))
    rho = rho.round(3).astype(str) + p

    return rho


