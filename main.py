import matplotlib.pyplot as plt
from sympy import *
import numpy as np
from cmath import isnan
from tqdm import tqdm


ALPHA = 0.2
THETA = 0.2
B = 1.2
GAMMA = 1.38
C_0_CONST = 0.5213
L_0_CONST = 1.328
N = 10000


# двумерная OLG система
def f(c, l):
    return pow(pow(l, GAMMA) - pow(c, THETA), 1 / ALPHA), B * (l - c)


# точки покоя
def equilibrium():
    c = symbols('c')
    x = np.array(solve(c ** ALPHA - (B * c / (B - 1)) ** GAMMA + c ** THETA))
    y = B * x / (B - 1)
    return x, y


def plot(c_0=C_0_CONST, l_0=L_0_CONST):
    C = []
    L = []
    #plt.scatter(c_0, l_0, s=10, color='red')
    for _ in range(N):
        c_0, l_0 = f(c_0, l_0)
        print(c_0, l_0)
        C.append(c_0)
        L.append(l_0)
    plt.scatter(C, L, s=1, edgecolors='blue')
    plt.xlabel('$с$')
    plt.ylabel('$l$')
    #plt.title("$\\alpha = {2}, \\theta = {3}, c = {4}, l = {5}, \\gamma = {0}$, N = {1}".format(GAMMA, N, ALPHA, THETA, C_0_CONST, L_0_CONST))
    #plt.show()
    #plt.savefig('C:\\Users\\user\\Desktop\\SPBU\\magistracy\\diploma\\fig\\OLG2mapHaoticAttractor.pdf')


def to_nan(c_0, l_0):
    n = 100
    for _ in range(n):
        c_0, l_0 = f(c_0, l_0)
        if isnan(c_0) or isnan(l_0) or c_0.imag != 0 or l_0.imag != 0:
            return True
    return False


def pool(C, L):
    vectorize_to_nan = np.vectorize(to_nan)
    pool = list()
    for y in L:
        pool.append(vectorize_to_nan(C, y))
    return pool


def plot_scatter(C, L, is_nan):
    if is_nan:
        plt.scatter(C, L, s=1, color='black')
    else:
        plt.scatter(C, L, s=1, color='red')


def plot_pool(c_min, c_max, l_min, l_max, n):
    C = np.linspace(c_min, c_max, num=n)
    L = np.linspace(l_min, l_max, num=n)
    mas_pool = pool(C, L)
    vect_plot_scatter = np.vectorize(plot_scatter)
    for i, y in tqdm(enumerate(L)):
        vect_plot_scatter(C, y, mas_pool[i])


plot_pool(-0.5, 1.5, -0.5, 2, 100)
plot()
plt.show()
