import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.interpolate import UnivariateSpline

def plot_solution_slice(x, V_slice, smooth=True, s_param=1e-2):
    """
    Trace la solution à un instant donné.
    
    Arguments :
    x        : array, grille spatiale
    V_slice  : array, valeurs de V à un instant fixe
    smooth   : bool, si True, trace une ligne approximante lisse
    s_param  : float, paramètre de lissage pour UnivariateSpline
    """
    fig, ax = plt.subplots(figsize=(8,5))
    ax.plot(x, V_slice, 'o', alpha=0.3, label='Solution brute')  # points bruts

    if smooth:
        spline = UnivariateSpline(x, V_slice, s=s_param)
        y_smooth = spline(x)
        ax.plot(x, y_smooth, '-', lw=2, label='Ligne approximante')

    ax.set_xlabel("S")
    ax.set_ylabel("V")
    ax.grid(True)
    ax.legend()
    return fig

def plot_surface(x, t, V, title="Surface V(t,x)"):
    """Trace la surface V(t,x)"""
    X, T = np.meshgrid(x, t)
    fig = plt.figure(figsize=(7,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, T, V, cmap='viridis')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_zlabel("V")
    ax.set_title(title)
    return fig

def plot_heatmap(x, t, V, title="Heatmap V(t,x)"):
    """Trace une heatmap de V(t,x)"""
    fig, ax = plt.subplots(figsize=(7,4))
    c = ax.pcolormesh(x, t, V, shading='auto', cmap='viridis')
    ax.set_xlabel("x")
    ax.set_ylabel("t")
    ax.set_title(title)
    fig.colorbar(c, ax=ax)
    return fig

def plot_sensitivity(x, solutions, param_values, param_name="sigma"):
    """Trace plusieurs solutions pour différentes valeurs d'un paramètre"""
    fig, ax = plt.subplots(figsize=(6,4))
    for V_slice, val in zip(solutions, param_values):
        ax.plot(x, V_slice, lw=2, label=f"{param_name}={val:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("V(t=0,x)")
    ax.set_title(f"Sensibilité de V à {param_name}")
    ax.grid(True)
    ax.legend()
    return fig
