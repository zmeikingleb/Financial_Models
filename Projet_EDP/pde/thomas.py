import numpy as np

def thomas(a, b, c, d):
    # Résout un système tridiagonal Ax = d
    # a : sous-diagonale (taille n-1)
    # b : diagonale (taille n)
    # c : sur-diagonale (taille n-1)
    # d : second membre (taille n)

    n = len(b)
    c_ = np.zeros(n-1)
    d_ = np.zeros(n)

    # Forward elimination
    if abs(b[0]) < 1e-14:
        raise ValueError("Pivot nul en i=0 (matrice non stable)")
    c_[0] = c[0] / b[0]
    d_[0] = d[0] / b[0]

    for i in range(1, n-1):
        temp = b[i] - a[i-1] * c_[i-1]
        if abs(temp) < 1e-14:
            raise ValueError(f"Pivot trop petit en i={i}, vérifier dx/dt et barrières")
        c_[i] = c[i] / temp
        d_[i] = (d[i] - a[i-1] * d_[i-1]) / temp

    # Dernière ligne
    temp = b[-1] - a[-1] * c_[-1]
    if abs(temp) < 1e-14:
        raise ValueError("Pivot nul en dernière ligne")
    d_[-1] = (d[-1] - a[-1] * d_[-2]) / temp

    # Backward substitution
    x = np.zeros(n)
    x[-1] = d_[-1]
    for i in reversed(range(n-1)):
        x[i] = d_[i] - c_[i] * x[i+1]

    return x