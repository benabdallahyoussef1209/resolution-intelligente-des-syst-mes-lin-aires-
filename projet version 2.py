import numpy as np

# =========================
# Données (tu peux changer)
# =========================
A = np.array([[2, 1, 0, 0],
              [1, 3, 1, 0],
              [0, 1, 4, 1],
              [0, 0, 1, 2]], dtype=float)

b = np.array([4, 10, 18, 11], dtype=float)
# =========================
# Vérifications
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
    v_p = np.linalg.eigvals(A)
    return np.all(v_p > 0)

def choose_method(A):
    n = len(A)

    # Vérifications de base
    if A.shape[0] != A.shape[1]:
        return "Erreur : matrice non carrée"

    if abs(np.linalg.det(A)) < 1e-12:
        return "Erreur : det(A)=0"

    # 1️⃣ Gauss-Seidel si diagonale dominante
    if is_diagonally_dominant(A):
        return "Gauss-Seidel"

    # 2️⃣ Gauss-Seidel si SPD
    if is_symmetric(A) and is_positive_definite(A):
        return "Gauss-Seidel"

    # 3️⃣ Cramer seulement petit système
    if n <= 3:
        return "Cramer"

    # 4️⃣ Sinon LU
    return "LU"

# =========================
# 1. CRAMER
# =========================
def cramer(A, b):
    n = len(b)
    detA = np.linalg.det(A)
    x = np.zeros(n)

    for i in range(n):
        Ai = A.copy()
        Ai[:, i] = b
        x[i] = np.linalg.det(Ai) / detA

    return x

# =========================
# 2. LU
# =========================
import numpy as np  # bibliothèque numérique

def lu_decomposition(A):
    A = np.array(A, dtype=float)
    n = len(A) 

    L = np.eye(n)  
    U = np.zeros((n, n)) 

    for i in range(n): 

    
        for k in range(i, n):
            U[i, k] = A[i, k] - sum(L[i, j]*U[j, k] for j in range(i))  
    
        for k in range(i+1, n):
            L[k, i] = (A[k, i] - sum(L[k, j]*U[j, i] for j in range(i))) / U[i, i] 


    return L, U  

def LU(A, b):
    L, U = lu_decomposition(A)  

    b = np.array(b, dtype=float)  
    n = len(b)

    y = np.zeros(n)
    for i in range(n):
        y[i] = b[i] - sum(L[i, j]*y[j] for j in range(i))
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - sum(U[i, j]*x[j] for j in range(i+1, n))) / U[i, i]

    return x 


# =========================
# 3. GAUSS-SEIDEL
# =========================
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

    # Affiche uniquement la solution finale
    print("Solution finale :", x)
    return x


# =========================
# PROGRAMME PRINCIPAL
# =========================
print("="*50)
print("Matrice A :\n", A)
print("Vecteur b :", b)
print("="*50)

method = choose_method(A)

print(f"\n👉 Méthode choisie : {method}")

if method == "Cramer":
    x = cramer(A, b)
    print("\nSolution (Cramer) :")
    print(x)

elif method == "LU":
    x = LU (A, b)
    print("\nSolution (LU) :")
    print(x)

elif method == "Gauss-Seidel":
    x = gauss_seidel(A, b)
    print("\nSolution finale :")
    print(x)

else:
    print(method)