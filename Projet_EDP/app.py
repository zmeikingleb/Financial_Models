import streamlit as st
from models.black_scholes import BlackScholesModel
from models.vasicek import VasicekModel
from models.cir import CIRModel
from payoffs.option_payoff import OptionPayoff
from pde.solver import PDESolver
from plots.plotting import plot_solution_slice, plot_surface, plot_heatmap
import matplotlib.pyplot as plt

# Configuration de la page Streamlit
st.set_page_config(page_title="EDP Finance", layout="wide")

# Titre principal
st.title("Résolution d’EDP en Finance")
st.markdown("Solveur générique par **Crank–Nicolson + Thomas**")

# Choix du modèle
model_choice = st.sidebar.selectbox("Modèle", ["Black-Scholes", "Vasicek", "CIR"])

# Paramètres numériques communs
tmax = st.sidebar.slider("Maturité T", 0.01, 5.0, 0.25)
Nx = st.sidebar.slider("Nx", 50, 300, 101)
Nt = st.sidebar.slider("Nt", 100, 2000, 500)

# Si Black-Scholes
if model_choice == "Black-Scholes":
    # Paramètres BS
    K = st.sidebar.number_input("Strike K", 1, 500, 100)
    sigma = st.sidebar.slider("Sigma", 0.01, 1.0, 0.2)
    r = st.sidebar.slider("r", 0.0, 0.2, 0.08)
    q = st.sidebar.slider("Dividende q", 0.0, 0.2, 0.0)

    # Type d'option
    option_style = st.sidebar.selectbox("Type d'option", ["Vanilla", "Exotic"])
    if option_style == "Vanilla":
        option_type = st.sidebar.selectbox("Option type", ["call", "put"])
        option_payoff = OptionPayoff(K, option_type=option_type, exotic=False)
        xmin, xmax = 0.0, 3*K
    else:
        exotic_choice = st.sidebar.selectbox("Type d'option exotique", ["Barrier", "Knockout"])
        option_type = st.sidebar.selectbox("Option type", ["call", "put"])
        kwargs = {}
        if exotic_choice == "Barrier":
            barrier_type = st.sidebar.selectbox("Type de barrière", ["down", "up", "double"])
            if barrier_type == "down":
                barrier = st.sidebar.number_input("Down barrier", 0.0, 500.0, 80.0)
                xmin, xmax = barrier, 2*K
            elif barrier_type == "up":
                barrier = st.sidebar.number_input("Up barrier", 0.0, 500.0, 150.0)
                xmin, xmax = 0.0, barrier
            else:
                B_low = st.sidebar.number_input("Barrier basse", 0.0, 500.0, 80.0)
                B_high = st.sidebar.number_input("Barrier haute", 0.0, 500.0, 150.0)
                xmin, xmax = B_low, B_high
                barrier = (B_low, B_high)
            kwargs['barrier'] = barrier
            kwargs['barrier_type'] = barrier_type
        elif exotic_choice == "Knockout":
            knockout_barrier = st.sidebar.number_input("Knockout barrier", 0.0, 500.0, 120.0)
            kwargs['knockout'] = True
            kwargs['knockout_barrier'] = knockout_barrier
            xmin, xmax = 0.0, knockout_barrier if option_type=="call" else 2*K

        option_payoff = OptionPayoff(K, option_type=option_type, exotic=True, **kwargs)

    # Création modèle et solveur BS
    model = BlackScholesModel(K, sigma, r, q, option_payoff)
    theta_scheme = st.sidebar.slider("Theta (0=Explicit,0.5=CN,1=Implicit)", 0.0, 1.0, 0.5)
    solver = PDESolver(model, xmin, xmax, Nx, tmax=tmax, Nt=Nt, theta=theta_scheme)
    V, x_grid, t_grid = solver.solve()

    # Graphiques BS
    st.subheader("V(t=0,x)")
    fig1 = plot_solution_slice(x_grid, V[0, :])
    st.pyplot(fig1)

    st.subheader("Surface V(t,x)")
    fig2 = plot_surface(x_grid, t_grid, V)
    st.pyplot(fig2)

    st.subheader("Heatmap V(t,x)")
    fig3 = plot_heatmap(x_grid, t_grid, V)
    st.pyplot(fig3)

# Si Vasicek ou CIR
else:
    # Choix des paramètres communs pour Vasicek et CIR
    if model_choice == "Vasicek":
        a = st.sidebar.slider("Vitesse de réversion a", 0.01, 2.0, 0.95)
        b = st.sidebar.slider("Niveau moyen b", -1.0, 1.0, 0.10)
        sigma = st.sidebar.slider("Volatilité sigma", 0.01, 1.0, 0.2)
        lam = st.sidebar.slider("Lambda", 0.0, 0.2, 0.05)
        xmin, xmax = -1.0, 1.0
    else:
        # CIR
        kappa = st.sidebar.slider("Vitesse de réversion kappa", 0.01, 2.0, 0.8)
        theta_param = st.sidebar.slider("Niveau moyen theta", 0.0, 1.0, 0.1)
        sigma = st.sidebar.slider("Volatilité sigma", 0.01, 1.0, 0.5)
        lam = st.sidebar.slider("Lambda", 0.0, 0.2, 0.05)
        xmin, xmax = 0.0, 1.0

    # Payoff vanilla sur le taux
    option_type = st.sidebar.selectbox("Option type", ["call", "put"])
    K = st.sidebar.number_input("Strike K", xmin, xmax, (xmin+xmax)/2)
    option_payoff = OptionPayoff(K, option_type=option_type, exotic=False)

    # Choix des theta à comparer
    theta_list = st.sidebar.multiselect(
        "Valeurs de theta à comparer",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
        default=[0.0, 0.5, 1.0]
    )

    # Création du modèle selon le choix
    if model_choice == "Vasicek":
        model = VasicekModel(a, b, sigma, lam, payoff=option_payoff)
    else:
        model = CIRModel(kappa, theta_param, sigma, lam, payoff=option_payoff)

    # Graphique comparatif pour différents theta
    st.subheader(f"Comparaison de V(t=0,r) pour différents θ ({model_choice})")
    plt.figure(figsize=(10,6))
    for theta in theta_list:
        solver = PDESolver(model, xmin, xmax, Nx, tmax=tmax, Nt=Nt, theta=theta)
        V, x_grid, t_grid = solver.solve()
        plt.plot(x_grid, V[0,:], label=f"θ={theta}")
    plt.xlabel("Taux r")
    plt.ylabel("Valeur V(t=0,r)")
    plt.title(f"Impact de θ sur la solution PDE {model_choice}")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

    # Surface et heatmap pour θ=0.5 par défaut
    solver_default = PDESolver(model, xmin, xmax, Nx, tmax=tmax, Nt=Nt, theta=0.5)
    V_default, x_grid, t_grid = solver_default.solve()

    st.subheader(f"Surface V(t,r) pour θ=0.5 ({model_choice})")
    fig_surface = plot_surface(x_grid, t_grid, V_default)
    st.pyplot(fig_surface)

    st.subheader(f"Heatmap V(t,r) pour θ=0.5 ({model_choice})")
    fig_heatmap = plot_heatmap(x_grid, t_grid, V_default)
    st.pyplot(fig_heatmap)
