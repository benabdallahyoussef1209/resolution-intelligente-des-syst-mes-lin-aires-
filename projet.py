import numpy as np
# 🔍 Vérifications de base
def est_carree(A):
    return A.shape[0] == A.shape[1]

def det_non_nul(A):
    return np.linalg.det(A) != 0

def diagonale_dominante(A):
    n = A.shape[0]
    for i in range(n):
        somme = sum(abs(A[i][j]) for j in range(n) if j != i)
        if abs(A[i][i]) <= somme:
            return False
    return True

def symetrique_definie_positive(A):
    if not np.allclose(A, A.T):
        return False
    eigenvalues = np.linalg.eigvals(A)
    return np.all(eigenvalues > 0)

# -----------------------------
# ⚙️ Méthodes de résolution
# -----------------------------

def cramer(A, b):
    n = len(b)
    det_A = np.linalg.det(A)
    x = np.zeros(n)
    for i in range(n):
        A_temp = A.copy()
        A_temp[:, i] = b
        x[i] = np.linalg.det(A_temp) / det_A
    return x

def lu_solution(A, b):
    return np.linalg.solve(A, b)

def gauss_seidel(A, b, tol=1e-10, max_iter=1000):
    n = len(b)
    x = np.zeros(n)
    for k in range(max_iter):
        x_new = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]
        if np.linalg.norm(x_new - x, ord=np.inf) < tol:
            return x_new, k
        x = x_new
    return x, max_iter

# -----------------------------
# 🧠 Choix automatique
# -----------------------------

def solve_system(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)

    if not est_carree(A):
        return "❌ Matrice non carrée"
    if not det_non_nul(A):
        return "❌ det(A) = 0 → pas de solution unique"

    n = A.shape[0]

    # Petit système → Cramer
    if n <= 3:
        print("👉 Méthode choisie : Cramer")
        return cramer(A, b)

    # Diagonale dominante → Gauss-Seidel
    if diagonale_dominante(A):
        print("👉 Méthode choisie : Gauss-Seidel (diagonale dominante)")
        sol, it = gauss_seidel(A, b)
        print("Nombre d'itérations :", it)
        return sol

    # Matrice SPD → Gauss-Seidel
    if symetrique_definie_positive(A):
        print("👉 Méthode choisie : Gauss-Seidel (SPD)")
        sol, it = gauss_seidel(A, b)
        print("Nombre d'itérations :", it)
        return sol

    # Cas général → LU
    print("👉 Méthode choisie : LU")
    return lu_solution(A, b)

# -----------------------------
# 🧪 Exemple d'utilisation
# -----------------------------

A = [[4, 1, 2],
     [1, 3, 1],
     [2, 1, 5]]

b = [4, 5, 6]

solution = solve_system(A, b)
print("Solution :", solution)