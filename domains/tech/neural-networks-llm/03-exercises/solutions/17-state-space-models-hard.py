"""
Solutions HARD — Jour 17 : State Space Models
=============================================
Exercices 7, 8, 9 (hard). Pur NumPy, comme 02-code/17-state-space-models.py.
Chaque etape non triviale est commentee avec le POURQUOI.

Run: python 03-exercises/solutions/17-state-space-models-hard.py
"""

import sys
import io
import numpy as np

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

np.random.seed(17)


# ============================================================================
# Exercice 7 — Parallel scan (Blelloch / Hillis-Steele)
# ============================================================================

def scan_op(left, right):
    """
    Operateur associatif de composition de deux segments affines.
    Un segment (a, b) represente la transformation h -> a*h + b.
    Composer (a1,b1) PUIS (a2,b2) :
        h -> a2*(a1*h + b1) + b2 = (a2*a1)*h + (a2*b1 + b2)
    donc (a1,b1) o (a2,b2) = (a1*a2, a2*b1 + b2).
    """
    a1, b1 = left
    a2, b2 = right
    return (a1 * a2, a2 * b1 + b2)


def sequential_scan(a, b):
    """Reference : h_t = a_t * h_{t-1} + b_t, h_0 = 0. Renvoie tous les h_t."""
    N = len(a)
    h = np.zeros(N)
    prev = 0.0
    for t in range(N):
        prev = a[t] * prev + b[t]
        h[t] = prev
    return h


def parallel_scan(a, b):
    """
    Scan inclusif via Hillis-Steele (doublement de pas). A chaque etage, on
    compose element[i] avec element[i-d]. O(N log N) work, O(log N) depth.
    Le resultat h_t est le 'b' du tuple prefixe cumule (car h_0 = 0).
    """
    N = len(a)
    A = a.astype(np.float64).copy()
    Bv = b.astype(np.float64).copy()
    d = 1
    while d < N:
        A_new = A.copy()
        B_new = Bv.copy()
        for i in range(d, N):
            # prefixe[i] = prefixe[i-d] o element[i] (compose dans l'ordre temporel)
            a_comp, b_comp = scan_op((A[i - d], Bv[i - d]), (A[i], Bv[i]))
            A_new[i], B_new[i] = a_comp, b_comp
        A, Bv = A_new, B_new
        d *= 2
    return Bv   # h_t = composante 'b' du prefixe cumule (h_0 = 0)


def exercice_7():
    print("=" * 70)
    print("Exercice 7 : Parallel scan (associatif) = selective scan de Mamba")
    print("=" * 70)

    # 1) verifier l'associativite a la main sur un exemple.
    e1, e2, e3 = (0.5, 1.0), (0.9, 2.0), (0.7, -1.0)
    left = scan_op(scan_op(e1, e2), e3)
    right = scan_op(e1, scan_op(e2, e3))
    print(f"\n  Associativite : (e1 o e2) o e3 = {tuple(np.round(left, 6))}")
    print(f"                  e1 o (e2 o e3) = {tuple(np.round(right, 6))}")
    assert np.allclose(left, right)

    # 2) + 3) + 4) equivalence sequentiel vs parallele (coeffs fixes / stables).
    N = 64
    rng = np.random.default_rng(0)
    a = rng.uniform(0.1, 0.95, size=N)      # a_t dans (0,1) -> stable
    b = rng.standard_normal(N)
    h_seq = sequential_scan(a, b)
    h_par = parallel_scan(a, b)
    diff = float(np.max(np.abs(h_seq - h_par)))
    print(f"\n  N={N} coeffs fixes : max|h_seq - h_par| = {diff:.2e}")
    assert diff < 1e-10

    # 5) selectivite : a_t et b_t input-dependent (Mamba). Le scan est agnostique
    # a la provenance des coeffs -> il marche identiquement.
    def softplus(z):
        return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)   # stable

    x = rng.standard_normal(N)
    w_a, w_b = 0.8, 0.5
    a_sel = np.exp(-softplus(w_a * x))      # dans (0,1) : a_t = exp(-softplus(.))
    b_sel = (w_b * x) * x                    # b_t input-dependent
    h_seq_s = sequential_scan(a_sel, b_sel)
    h_par_s = parallel_scan(a_sel, b_sel)
    diff_s = float(np.max(np.abs(h_seq_s - h_par_s)))
    print(f"  N={N} coeffs SELECTIFS (input-dependent) : "
          f"max|h_seq - h_par| = {diff_s:.2e}")
    assert diff_s < 1e-10

    # 6) analyse.
    print("\n  POURQUOI l'associativite = condition du parallelisme : seul un")
    print("  operateur associatif permet de regrouper les compositions en ARBRE")
    print("  (combiner [1..4] = [1..2] o [3..4]) -> profondeur O(log N).")
    print("  POURQUOI Mamba NE peut PAS utiliser la FFT (contrairement a S4) : ses")
    print("  coeffs a_t = exp(Delta_t A) dependent de x_t -> kernel NON stationnaire")
    print("  -> pas de convolution unique -> pas de FFT. Le parallel scan, lui,")
    print("  marche quels que soient les coeffs (il ne suppose pas la stationnarite).")
    print("  Depth vs bandwidth : le parallel scan resout la PROFONDEUR (O(log N)")
    print("  au lieu de O(N)) ; le kernel SRAM de Tri Dao resout la BANDE PASSANTE")
    print("  (scan en SRAM + recompute, pas de materialisation de tous les h_t en HBM).")

    print("\n  [OK] Exercice 7")


# ============================================================================
# Exercice 8 — Discretisation ZOH et stabilite
# ============================================================================

def discretize(A_diag, B, Delta):
    """
    Discretisation Zero-Order Hold pour A DIAGONALE (vecteur des valeurs propres) :
      A_bar = exp(Delta * A)                       (element-wise sur la diagonale)
      B_bar = (A_bar - I) @ inv(A) @ B = ((A_bar - 1) / A) * B   (diagonale)
    On gere le cas diagonal directement (pas d'expm de matrice pleine).
    """
    A_bar = np.exp(Delta * A_diag)                 # (D,) exp des valeurs propres
    # (A_bar - 1)/A est la forme diagonale de (A_bar - I) A^{-1}.
    factor = (A_bar - 1.0) / A_diag                # A_diag < 0 -> bien defini
    B_bar = factor[:, None] * B                    # (D, 1)
    return A_bar, B_bar


def continuous_euler(A_diag, B, C, x, Delta, substeps=2000):
    """Integration fine (Euler) de h'(t)=A h + B x(t), x constant par pas (ZOH)."""
    D = len(A_diag)
    N = len(x)
    h = np.zeros(D)
    y = np.zeros(N)
    dt = Delta / substeps
    for t in range(N):
        for _ in range(substeps):
            h = h + dt * (A_diag * h + B.flatten() * x[t])
        y[t] = (C @ h).item()
    return y


def discrete_recurrence(A_bar, B_bar, C, x):
    """h_t = A_bar h_{t-1} + B_bar x_t ; y_t = C h_t."""
    D = len(A_bar)
    h = np.zeros(D)
    y = np.zeros(len(x))
    for t in range(len(x)):
        h = A_bar * h + B_bar.flatten() * x[t]
        y[t] = (C @ h).item()
    return y


def exercice_8():
    print("\n" + "=" * 70)
    print("Exercice 8 : Discretisation ZOH et role de Delta")
    print("=" * 70)

    rng = np.random.default_rng(1)
    # 1) systeme continu : A diagonale STABLE (valeurs propres negatives).
    A_diag = np.array([-0.1, -0.5, -1.0, -2.0])
    D = len(A_diag)
    B = rng.standard_normal((D, 1)) * 0.5
    C = rng.standard_normal((1, D)) * 0.5

    # 2) + 3) discretiser puis comparer a l'integration fine de l'EDO.
    Delta = 0.1
    A_bar, B_bar = discretize(A_diag, B, Delta)
    x = rng.standard_normal(20)
    y_disc = discrete_recurrence(A_bar, B_bar, C, x)
    y_cont = continuous_euler(A_diag, B, C, x, Delta, substeps=4000)
    diff = float(np.max(np.abs(y_disc - y_cont)))
    print(f"\n  Delta={Delta} : A_bar = {np.round(A_bar, 4).tolist()}")
    print(f"  max|recurrence discrete - EDO integree fin| = {diff:.2e}")
    print("  -> la discretisation ZOH reproduit bien le systeme continu.")

    # 4) effet de Delta : petit Delta -> retient longtemps ; grand -> oublie vite.
    print("\n  Effet de Delta (mode dominant, val. propre la plus lente = -0.1) :")
    print(f"    {'Delta':<8} {'A_bar[0]':<12} {'demi-vie (steps)':<18}")
    a_slow = A_diag[0]                              # -0.1
    half_lives = []
    for Delta in [0.01, 0.1, 1.0, 5.0]:
        ab = np.exp(Delta * a_slow)                # A_bar du mode lent
        # demi-vie : nb de steps n tels que ab^n = 0.5 -> n = ln(0.5)/ln(ab).
        hl = np.log(0.5) / np.log(ab) if ab < 1 else np.inf
        half_lives.append(hl)
        print(f"    {Delta:<8} {ab:<12.5f} {hl:<18.2f}")
    print("  -> petit Delta -> A_bar ~ 1 -> memoire LONGUE (demi-vie elevee) ;")
    print("     grand Delta -> A_bar ~ 0 -> memoire COURTE (oublie vite, absorbe).")

    # 5) lien Mamba : Delta_t input-dependent ouvre/ferme la memoire.
    print("\n  Delta_t input-dependent (Mamba) sur une mini-sequence :")
    # Token 'important' (|x| grand) -> grand Delta (absorbe) ; sinon petit Delta.
    x_seq = np.array([0.1, 0.1, 3.0, 0.1, 0.1])    # 1 token saillant au milieu
    Delta_t = 0.05 + 1.5 * np.abs(x_seq)           # softplus-like : grand si |x| grand
    h = np.zeros(D)
    print(f"    {'t':<3} {'x_t':<7} {'Delta_t':<9} {'|h| apres':<10}")
    for t in range(len(x_seq)):
        A_bar_t, B_bar_t = discretize(A_diag, B, Delta_t[t])
        h = A_bar_t * h + B_bar_t.flatten() * x_seq[t]
        print(f"    {t:<3} {x_seq[t]:<7.2f} {Delta_t[t]:<9.3f} "
              f"{np.linalg.norm(h):<10.4f}")
    print("  -> grand Delta_t sur le token saillant : la memoire 's'ouvre' et")
    print("     absorbe l'info ; petit Delta_t ailleurs : la memoire reste figee.")

    # 6) analyse.
    print("\n  POURQUOI exiger la stabilite : Re(eig(A))<0 en continu equivaut a")
    print("  |eig(A_bar)|<1 en discret (A_bar = exp(Delta A)). Sinon l'etat diverge.")
    print("  POURQUOI A fixe / Delta selectif : rendre A selectif coute cher et")
    print("  menace la stabilite + detruit l'init HiPPO (optimale pour le long-range).")
    print("  Delta_t selectif suffit : Ā = exp(Delta_t A) devient input-dependent,")
    print("  donc la selectivite agit via Delta tout en gardant A stable et expressif.")

    # --- assertions de validation ---
    assert diff < 1e-2                              # discret ~ continu
    assert np.all(np.abs(A_bar) < 1.0)             # discret stable
    # Demi-vie decroit quand Delta croit (memoire plus courte).
    assert half_lives[0] > half_lives[-1]
    print("\n  [OK] Exercice 8")


# ============================================================================
# Exercice 9 — Capacite memoire de l'etat : quand le compresseur sature
# ============================================================================

def write_memory(keys, vals):
    """Memoire associative lineaire : H = sum_m v_m k_m^T (etat D x D fixe)."""
    D = keys.shape[1]
    H = np.zeros((D, D))
    for k, v in zip(keys, vals):
        H += np.outer(v, k)
    return H


def recall_cosine(H, keys, vals):
    """Recall v_hat = H k_query ; cosine similarity moyenne avec le vrai v."""
    sims = []
    for k, v in zip(keys, vals):
        v_hat = H @ k
        cos = float(np.dot(v_hat, v) /
                    (np.linalg.norm(v_hat) * np.linalg.norm(v) + 1e-12))
        sims.append(cos)
    return float(np.mean(sims))


def exercice_9():
    print("\n" + "=" * 70)
    print("Exercice 9 : Capacite memoire de l'etat (compresseur lossy)")
    print("=" * 70)

    rng = np.random.default_rng(2)

    def make_pairs(M, D):
        keys = rng.standard_normal((M, D))
        keys /= np.linalg.norm(keys, axis=1, keepdims=True) + 1e-12  # cles unitaires
        vals = rng.standard_normal((M, D))
        return keys, vals

    # 2) + 3) degradation du recall SSM (etat D fixe) quand M croit.
    D = 16
    print(f"\n  Memoire SSM (etat {D}x{D} fixe), recall cosine vs nb de paires M :")
    print(f"    {'M':<6} {'recall cosine SSM':<20} {'recall transformer':<18}")
    Ms = [4, 8, 16, 32, 64, 128]
    cos_curve = []
    sat_M = None
    for M in Ms:
        keys, vals = make_pairs(M, D)
        H = write_memory(keys, vals)
        cos_ssm = recall_cosine(H, keys, vals)
        cos_curve.append(cos_ssm)
        # Transformer = KV cache : stocke chaque paire -> recall parfait.
        cos_tf = 1.0
        if sat_M is None and cos_ssm < 0.7:
            sat_M = M
        print(f"    {M:<6} {cos_ssm:<20.3f} {cos_tf:<18.3f}")
    print(f"  Point de saturation (recall < 0.7) ~ M = {sat_M} pour D = {D}")

    # 4) comparaison SSM (sature) vs transformer (tient).
    print("\n  -> SSM : memoire O(D) FIXE -> sature des que M depasse ~capacite.")
    print("     Transformer : memoire O(M) -> stocke chaque paire -> recall ~parfait")
    print("     (au cout d'une memoire qui croit lineairement avec M).")

    # 5) loi de capacite : M_saturation ~ lineaire en D.
    print("\n  Loi de capacite : M_saturation (recall < 0.7) vs D :")
    print(f"    {'D':<6} {'M_saturation':<14}")
    Ds = [8, 16, 32, 64]
    sat_points = []
    for Dd in Ds:
        sat = None
        for M in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            keys, vals = make_pairs(M, Dd)
            H = write_memory(keys, vals)
            if recall_cosine(H, keys, vals) < 0.7:
                sat = M
                break
        sat = sat if sat is not None else 512
        sat_points.append(sat)
        print(f"    {Dd:<6} {sat:<14}")
    print("  -> M_saturation croit ~lineairement avec D : un etat plus grand")
    print("     stocke proportionnellement plus de paires avant de saturer.")

    # 6) analyse.
    print("\n  POURQUOI c'est une limite FONDAMENTALE (theorie de l'information) :")
    print("  on compresse M*D floats (les paires) dans un etat de D*D < M*D scalaires.")
    print("  Au-dela, la compression est lossy par construction, pas par bug.")
    print("  Lien MQAR : Mamba pur ~30-50% vs transformer 95%+ (cours). Les hybrides")
    print("  (Jamba) sont la reponse pragmatique : rares layers d'attention pour le")
    print("  recall exact, layers Mamba pour le throughput et le long context.")

    # --- assertions de validation ---
    # Recall fort a faible M, degrade a fort M (etat D=16 fixe).
    assert cos_curve[0] > cos_curve[-1]
    assert cos_curve[-1] < 0.7                      # sature a M=128, D=16
    # Capacite croit avec D : M_saturation monotone non-decroissant.
    assert sat_points[-1] >= sat_points[0]
    print("\n  [OK] Exercice 9")


if __name__ == "__main__":
    exercice_7()
    exercice_8()
    exercice_9()
    print("\nDone (HARD).")
