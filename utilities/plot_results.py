from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_prediction(y1, y2, N_total, n_toplot=10**10,):
        
    idxs = np.arange(len(y1))
    np.random.shuffle(idxs)
    
    y_expected = y1.reshape(-1)[idxs[:n_toplot]]
    y_predicted = y2.reshape(-1)[idxs[:n_toplot]]

    xy = np.vstack([y_expected, y_predicted])
    z = gaussian_kde(xy)(xy)
    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    y_plt, ann_plt, z = y_expected[idx], y_predicted[idx], z[idx]
    
    fig = plt.figure(figsize=(6,6))
    plt.title("Model Evaluation", fontsize=17)
    plt.ylabel('Modeled SMB (m.w.e)', fontsize=16)
    plt.xlabel('Reference SMB (m.w.e)', fontsize=16)
    sc = plt.scatter(y_plt, ann_plt, c=z, s=20)
    plt.clim(0,0.4)
    plt.tick_params(labelsize=14)
    plt.colorbar(sc) 
    lineStart = -1.5
    lineEnd = 1.5
    plt.plot([lineStart, lineEnd], [lineStart, lineEnd], 'k-')
    plt.axvline(0.0, ls='-.', c='k')
    plt.axhline(0.0, ls='-.', c='k')
    plt.xlim(lineStart, lineEnd)
    plt.ylim(lineStart, lineEnd)
    plt.gca().set_box_aspect(1)
    
    textstr = '\n'.join((
    r'$R^2=%.2f$' % (r2_score(y_expected, y_predicted), ),
    r'$RMSE=%.2f$' % (mean_squared_error(y_expected, y_predicted), ),
    r'$N total=%.0f$' % (N_total), ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    #plt.show()

    return fig

def plot_smb(df_all):
    df_all_st = df_all.copy().dropna()
    y1 = df_all_st['OBS'].values
    y2 = df_all_st['SIM'].values
    textstr = '\n'.join((
    r'$R^2=%.2f$' % (r2_score(y1, y2), ),
    r'$RMSE=%.2f$' % (mean_squared_error(y1, y2), ),
    r'$N=%.0f$' % (len(y1)), ))
    fig, (ax0) = plt.subplots(figsize=(10,5)) 
    df_all.plot.bar(ax = ax0)
    ax0.set_xlabel('')
    ax0.set_ylabel(u'Surface Mass Balance (m w.e.)') 
    ax0.set_ylim(-1.0, 1.0)
    # Vamos agregar un caja para que se muestre los indicadores estadísticos. Aquí configuramos los propiedades de la caja
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # Ahora podemos agregar nuestros indicadores estadísticos
    plt.text(0.025, 0.25, textstr, transform=plt.gca().transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax0.axhline(0, linewidth=1, color='grey', linestyle ='-')

    return fig
