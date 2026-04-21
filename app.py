import streamlit as st
import numpy as np

# =========================
# 🔧 Fonctions mathématiques
# =========================

def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        if abs(A[i,i]) <= sum(abs(A[i,j]) for j in range(n) if j != i):
            return False
    return True

def is_symmetric(A):
    return np.allclose(A, A.T)

def is_positive_definite(A):
    return np.all(np.linalg.eigvals(A) > 0)

def cramer(A, b):
    n = len(b)
    detA = np.linalg.det(A)
    x = np.zeros(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA

    return x

def lu_decomposition(A):
    n = len(A)
    L = np.eye(n)
    U = np.zeros((n, n))

    for i in range(n):
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j] * U[j, k] for j in range(i))

        for k in range(i+1, n):
            L[k, i] = (A[k, i] - sum(L[k, j] * U[j, i] for j in range(i))) / U[i, i]

    return L, U

def lu_solve(A, b):
    L, U = lu_decomposition(A)
    n = len(b)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j] * y[j] for j in range(i))

    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i+1, n))) / U[i, i]

    return x

def gauss_seidel(A, b, tol=1e-6, max_iter=100):
    n = len(b)
    x = np.zeros(n)

    for _ in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x[i] = (b[i] - s) / A[i, i]

        if np.linalg.norm(x - x_old, ord=np.inf) < tol:
            break

    return x

def choose_method(A):
    n = len(A)

    if abs(np.linalg.det(A)) < 1e-12:
        return "Erreur : det(A)=0"

    if is_diagonally_dominant(A):
        return "Gauss-Seidel"

    if is_symmetric(A) and is_positive_definite(A):
        return "Gauss-Seidel"

    if n <= 3:
        return "Cramer"

    return "LU"

# =========================
# 🎨 STREAMLIT UI
# =========================

st.title("🔢 Résolution de systèmes linéaires")

st.write("Entrez une matrice A et un vecteur b")

# Taille du système
n = st.number_input("Taille du système (n)", min_value=2, max_value=10, value=3)

st.subheader("Matrice A")

A = []
for i in range(n):
    cols = st.columns(n)
    row = []
    for j in range(n):
        val = cols[j].number_input(f"A[{i+1},{j+1}]", value=0.0, key=f"a{i}{j}")
        row.append(val)
    A.append(row)

A = np.array(A, dtype=float)

st.subheader("Vecteur b")

b = []
cols = st.columns(n)
for i in range(n):
    val = cols[i].number_input(f"b[{i+1}]", value=0.0, key=f"b{i}")
    b.append(val)

b = np.array(b, dtype=float)

# =========================
# 🚀 Résolution
# =========================

if st.button("Résoudre"):

    method = choose_method(A)

    st.write("👉 Méthode choisie :", method)

    if method == "Cramer":
        x = cramer(A, b)
        st.success("Solution (Cramer)")
        st.write(x)

    elif method == "LU":
        x = lu_solve(A, b)
        st.success("Solution (LU)")
        st.write(x)

    elif method == "Gauss-Seidel":
        x = gauss_seidel(A, b)
        st.success("Solution (Gauss-Seidel)")
        st.write(x)

    else:
        st.error(method)