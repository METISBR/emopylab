# -*- coding: utf-8 -*-
"""
CMOEA-GBSS: Constrained Multi-Objective Evolutionary Algorithm with
Gradient-Based Stochastic Steepest Search

Algoritmo híbrido para otimização multi-objetivo com restrições que integra:
  1. Tri-arquivo (FA, DA, FEA) do CMOEA-CD para pressão de seleção especializada
  2. Vetores de referência uniformes + niching angular do NSGA-III para diversidade
  3. EEJ — Evolutionary Ensemble Jacobian: Jacobiano multi-objetivo estimado
     coletivamente sobre o arquivo FEA, sem custo adicional de avaliação
  4. Direção de descida comum via QP no simplex probabilístico (inspirado em SSW)
  5. Passo Euler-Maruyama estocástico para perturbação controlada

Paradigma: direction-driven + archive-driven + niching
  Geração → EEJ(FEA) → Variação (DIRECIONADA) → Seleção por arquivo → Nova geração

Inovação central (EEJ):
  Em vez de estimar um Jacobiano por trajetória individual (Broyden), o EEJ
  estima o Jacobiano da fronteira Pareto estimada usando TODOS os indivíduos
  não-dominados do FEA como amostras coletivas:

      J_hat = (ΔF)^T @ ΔX @ pinv(ΔX^T @ ΔX + reg·I)

  onde ΔX e ΔF são as variações centradas no arquivo FEA.
  Conecta com Ensemble Kalman Filter e estimação de gradiente zeroth-order.

Referências:
  - CMOEA-CD: Liu et al., IEEE TEC 2025, DOI 10.1109/TEVC.2024.3525153
  - NSGA-III: Deb & Jain, IEEE TEC 2014
  - EEJ / Zeroth-order: Nesterov & Spokoiny, Found. Comput. Math. 2017
  - QP-simplex: Fliege & Svaiter, Math. Methods Oper. Res. 2000

Autor: Research Team
Ano: 2026
"""

from __future__ import annotations

from typing import Optional, Dict

import numpy as np
from core.population import Population

from core.algorithm import Algorithm
from algorithms.community_utils.moead_family import (
    rng_from_algo,
    sample_initial,
)
from operators.utility_functions.OperatorDE import OperatorDE
from operators.utility_functions.OperatorGA import OperatorGA
from operators.utility_functions.TournamentSelection import TournamentSelection
from operators.utility_functions.UniformPoint import UniformPoint
from util.nds.non_dominated_sorting import NonDominatedSorting


# =============================================================================
# METADADOS DO ALGORITMO
# =============================================================================

ALGORITHM_FLAGS = {
    "CMOEA_GBSS": {"multi", "many", "constrained"},
}

ALGO_FLAGS      = {"multi", "many"}
OBJECTIVE_SCOPE = "many"


# =============================================================================
# UTILITÁRIOS INTERNOS
# =============================================================================

def _to_f(pop: Population) -> np.ndarray:
    """Extrai matriz de objetivos como float64 CPU."""
    return np.asarray(pop.get("F"), dtype=float)


def _cv(pop: Population) -> np.ndarray:
    """Calcula violação total de restrições (soma max(g,0))."""
    cv = pop.get("CV")
    if cv is not None:
        arr = np.asarray(cv, dtype=float).reshape(-1)
        if arr.size == len(pop):
            return np.maximum(arr, 0.0)
    g = pop.get("G")
    if g is None:
        return np.zeros(len(pop), dtype=float)
    g = np.asarray(g, dtype=float)
    if g.ndim == 1:
        g = g[:, None]
    return np.sum(np.maximum(g, 0.0), axis=1)


def _safe_norm(v: np.ndarray) -> float:
    """Norma vetorial com proteção anti-zero."""
    n = float(np.linalg.norm(v))
    return n if n > 1e-14 else 1.0


def _pairwise_dist(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distância euclidiana pairwise robusta."""
    AA = np.sum(A * A, axis=1, keepdims=True)
    BB = np.sum(B * B, axis=1, keepdims=True).T
    return np.sqrt(np.maximum(AA + BB - 2.0 * (A @ B.T), 0.0))


def _cosine_sim(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Similaridade cosseno pairwise entre linhas de A e B."""
    na = np.linalg.norm(A, axis=1, keepdims=True)
    nb = np.linalg.norm(B, axis=1, keepdims=True)
    denom = np.maximum(na * nb.T, 1e-32)
    return np.clip((A @ B.T) / denom, -1.0, 1.0)


def _normalize_F(F: np.ndarray, zmin: np.ndarray, zmax: np.ndarray) -> np.ndarray:
    """Normaliza objetivos para [0,1] com proteção de span nulo."""
    span = zmax - zmin
    span = np.where(np.abs(span) <= 1e-12, 1.0, span)
    return (F - zmin) / span


def _crowding_distance(F: np.ndarray) -> np.ndarray:
    """Distância de crowding padrão NSGA-II."""
    n, m = F.shape
    if n == 0:
        return np.empty(0, dtype=float)
    if n <= 2:
        return np.full(n, np.inf, dtype=float)
    cd = np.zeros(n, dtype=float)
    for k in range(m):
        idx = np.argsort(F[:, k])
        cd[idx[0]] = np.inf
        cd[idx[-1]] = np.inf
        denom = F[idx[-1], k] - F[idx[0], k]
        if denom <= 1e-32:
            continue
        interior = idx[1:-1]
        cd[interior] += (F[idx[2:], k] - F[idx[:-2], k]) / denom
    return cd


def _truncation_select(F: np.ndarray, k_keep: int) -> np.ndarray:
    """Seleciona k_keep índices por truncação (maior distância mínima ao vizinho)."""
    n = len(F)
    if k_keep >= n:
        return np.arange(n, dtype=int)
    k_del = n - k_keep
    D = _pairwise_dist(F, F)
    np.fill_diagonal(D, np.inf)
    deleted = np.zeros(n, dtype=bool)
    for _ in range(k_del):
        remain = np.where(~deleted)[0]
        sub = D[np.ix_(remain, remain)]
        temp = np.sort(sub, axis=1)
        rank = np.lexsort(temp.T[::-1])
        deleted[remain[rank[0]]] = True
    return np.where(~deleted)[0]


def _nsga3_niching_select(
    F: np.ndarray,
    W: np.ndarray,
    zmin: np.ndarray,
    k_keep: int,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Seleção por niching angular estilo NSGA-III."""
    n = len(F)
    if k_keep >= n:
        return np.arange(n, dtype=int)
    Fn = _normalize_F(F, zmin, np.max(F, axis=0))
    cos_wf = _cosine_sim(W, Fn)
    sin_wf = np.sqrt(np.maximum(1.0 - cos_wf * cos_wf, 0.0))
    dist_to_ref = np.linalg.norm(Fn, axis=1)
    perp = sin_wf * dist_to_ref[None, :]          # (n_ref, n)
    assoc = np.argmin(perp, axis=0)               # (n,) - ref de cada ponto
    # Seleciona k_keep pontos: rotaciona por niche com menor população
    chosen = []
    niche_count = np.zeros(len(W), dtype=int)
    available = np.ones(n, dtype=bool)
    # Usa rng controlado se fornecido, senão numpy global como fallback seguro
    _rng = rng if rng is not None else np.random.default_rng()
    while len(chosen) < k_keep:
        active_niches = np.where(
            np.any(available[None, :] & (assoc[None, :] == np.arange(len(W))[:, None]), axis=1)
        )[0]
        if active_niches.size == 0:
            break
        min_count = niche_count[active_niches].min()
        tied = active_niches[niche_count[active_niches] == min_count]
        # CORRIGIDO: usa rng.integers() em vez de np.random.randint()
        j = int(tied[int(_rng.integers(0, len(tied)))])
        cands = np.where(available & (assoc == j))[0]
        if cands.size == 0:
            continue
        pick = int(cands[int(np.argmin(perp[j, cands]))])
        chosen.append(pick)
        available[pick] = False
        niche_count[j] += 1
    # fallback se ainda faltarem slots
    if len(chosen) < k_keep:
        remain = np.where(available)[0]
        extra = k_keep - len(chosen)
        chosen.extend(remain[:extra].tolist())
    return np.asarray(chosen, dtype=int)


def _constraint_pareto_nondominated(
    pop: Population,
    add_constraint_rule: bool,
) -> np.ndarray:
    """
    Dominância Constraint-Pareto.
    Retorna máscara booleana dos não-dominados.
    """
    F = np.round(np.asarray(_to_f(pop), dtype=float), 10)
    cv = _cv(pop)
    n = len(F)
    dominated = np.zeros(n, dtype=bool)

    for i in range(n - 1):
        if dominated[i]:
            continue
        rem = np.arange(i + 1, n)
        rem = rem[~dominated[rem]]
        if rem.size == 0:
            continue

        diff      = F[i] - F[rem]
        max_err   = np.max(diff, axis=1)
        min_err   = np.min(diff, axis=1)
        eq        = np.all(diff == 0.0, axis=1)

        if not add_constraint_rule:
            dom_i = (~eq) & (min_err >= 0.0)
            dom_j = eq | ((~eq) & ~(min_err >= 0.0) & (max_err <= 0.0))
        else:
            cvi   = cv[i]
            cvj   = cv[rem]
            dom_i = (eq & (cvi > cvj)) | (
                (~eq) & (min_err >= 0.0) & ((cvj <= 0.0) | (cvj <= cvi))
            )
            dom_j = (eq & (cvi <= cvj)) | (
                (~eq) & ~(min_err >= 0.0) & (max_err <= 0.0) & ((cvi <= 0.0) | (cvi <= cvj))
            )

        hit = np.flatnonzero(dom_i)
        cut = int(hit[0]) if hit.size > 0 else rem.size
        if cut > 0 and np.any(dom_j[:cut]):
            dominated[rem[:cut][dom_j[:cut]]] = True
        if hit.size > 0:
            dominated[i] = True

    return ~dominated


# =============================================================================
# EEJ — EVOLUTIONARY ENSEMBLE JACOBIAN
# =============================================================================

def _compute_eej(
    X_archive: np.ndarray,
    F_archive: np.ndarray,
    reg: float = 1e-6,
) -> np.ndarray:
    """
    Estima o Jacobiano multi-objetivo coletivamente sobre um arquivo de soluções.

    O EEJ (Evolutionary Ensemble Jacobian) é fundamentalmente diferente do
    Broyden individual: em vez de acompanhar a trajetória de um único indivíduo,
    usa TODOS os indivíduos não-dominados do arquivo FEA como amostras coletivas
    para estimar o Jacobiano local da fronteira Pareto estimada.

    Formulação via mínimos quadrados regularizados:
        dX = X - mean(X)          (n_pop, n_var)  — variações de decisão
        dF = F - mean(F)          (n_pop, n_obj)  — variações de objetivo

        J_hat = (dF^T @ dX) @ pinv(dX^T @ dX + reg·I)

    Sob condições suaves (F Lipschitz, distribuição evolutiva não-degenerada):
        E[J_hat] → J_f(μ_FEA)   quando n_pop → ∞

    onde J_f é o Jacobiano real avaliado no centroide do arquivo.
    Conecta com estimação de gradiente zeroth-order (Nesterov & Spokoiny, 2017)
    e com Ensemble Kalman Filter (Evensen, 1994).

    Args:
        X_archive : matr. de decisão do arquivo (n_pop, n_var)
        F_archive : matr. de objetivos do arquivo (n_pop, n_obj)
        reg       : regularização Tikhonov para estabilidade numérica

    Returns:
        J_hat: Jacobiano ensemble, forma (n_obj, n_var)
    """
    n_pop, n_var = X_archive.shape
    n_obj        = F_archive.shape[1]

    if n_pop < 2:
        # Impossível estimar com 1 ponto — retorna zero
        return np.zeros((n_obj, n_var))

    # Centralizar: remove tendência linear da distribuição
    dX = X_archive - X_archive.mean(axis=0)   # (n_pop, n_var)
    dF = F_archive - F_archive.mean(axis=0)   # (n_pop, n_obj)

    # Gram matrix regularizada: G = dX^T @ dX + reg*I  → (n_var, n_var)
    G = dX.T @ dX + reg * np.eye(n_var)

    # Resolver as n_obj equações lineares simultâneas via lstsq:
    #   G @ J_hat^T = dX^T @ dF  →  J_hat = (dF^T @ dX) @ G^{-1}
    try:
        # Solução via lstsq: mais estável que inversa direta
        Jt, _, _, _ = np.linalg.lstsq(
            G,                  # (n_var, n_var)
            dX.T @ dF,          # (n_var, n_obj)
            rcond=None,
        )
        J_hat = Jt.T            # (n_obj, n_var)
    except np.linalg.LinAlgError:
        J_hat = np.zeros((n_obj, n_var))

    return J_hat


# =============================================================================
# DIREÇÃO DE DESCIDA COMUM VIA QP NO SIMPLEX
# =============================================================================

def _compute_common_descent(J: np.ndarray, reg: float = 1e-8) -> np.ndarray:
    """
    Calcula a direção de descida comum via QP no simplex probabilístico.

    Resolve: min_{α ∈ Δ}  α^T (J J^T + reg·I) α
    onde    Δ = {α ≥ 0 : Σα_i = 1}

    A direção resultante é:  q = J^T α* / ||J^T α*||

    Args:
        J:   Jacobiano aproximado, forma (n_obj, n_var)
        reg: Regularização para estabilidade numérica

    Returns:
        Vetor normalizado q de forma (n_var,)
    """
    m, n = J.shape

    if m == 1:
        q    = J[0]
        norm = float(np.linalg.norm(q))
        return q / norm if norm > 1e-12 else np.zeros(n)

    # Gram matrix regularizada
    H    = J @ J.T + reg * np.eye(m)
    L    = float(np.linalg.norm(H, ord=np.inf))
    step = 1.0 / (L + 1e-12)

    # Descida de gradiente projetada no simplex
    alpha = np.full(m, 1.0 / m)
    for _ in range(300):
        g         = H @ alpha
        alpha_new = alpha - step * g
        alpha_new = np.maximum(alpha_new, 0.0)
        s         = float(np.sum(alpha_new))
        if s < 1e-12:
            alpha_new = np.full(m, 1.0 / m)
        else:
            alpha_new /= s
        if float(np.linalg.norm(alpha_new - alpha)) < 1e-10:
            alpha = alpha_new
            break
        alpha = alpha_new

    q    = J.T @ alpha
    norm = float(np.linalg.norm(q))
    return q / norm if norm > 1e-12 else np.zeros(n)


# =============================================================================
# CLASSE PRINCIPAL
# =============================================================================

class CMOEA_GBSS(Algorithm):
    """
    CMOEA-GBSS: Constrained MOEA with Gradient-Based Stochastic Steepest Search.

    Combina:
      • Tri-arquivo (FA, DA, FEA) do CMOEA-CD → pressão de seleção especializada
      • Niching angular com vetores de referência uniformes → diversidade many-obj
      • Jacobiano Broyden por indivíduo → direção de descida sem custo extra
      • QP-simplex → direção comum a todos os objetivos
      • Euler-Maruyama estocástico → perturbação controlada no espaço de decisão
      • DE + GA como operadores de backup → robustez evolutiva
    """

    ALGO_FLAGS      = {"multi", "many"}
    OBJECTIVE_SCOPE = "many"

    def __init__(
        self,
        pop_size     : int   = 100,
        ref_dirs                  = None,
        step_size    : float = 0.05,
        noise_scale  : float = 0.02,
        prob_grad    : float = 0.40,
        e1           : int   = 1,
        e2           : int   = 1,
        de_cr        : float = 1.0,
        de_f         : float = 0.5,
        sampling                  = None,
        **kwargs,
    ):
        """
        Parâmetros
        ----------
        pop_size   : tamanho da população (= tamanho de cada arquivo FEA)
        ref_dirs   : vetores de referência (gerados automaticamente se None)
        step_size  : passo λ do Euler-Maruyama para descida de gradiente
        noise_scale: escala ε do ruído estocástico
        prob_grad  : probabilidade de usar operador gradiente vs. DE/GA
        e1         : modo de seleção no FA (1=truncação, 2=CD, 3=NSGA-3)
        e2         : modo de seleção no FEA (1=truncação, 2=CD, 3=NSGA-3)
        de_cr      : crossover rate do DE
        de_f       : fator de escala do DE
        sampling   : estratégia de amostragem inicial
        """
        super().__init__(**kwargs)

        self.pop_size    = int(max(pop_size, 4))
        self.ref_dirs    = None if ref_dirs is None else np.asarray(ref_dirs, dtype=float)
        self.step_size   = float(step_size)
        self.noise_scale = float(noise_scale)
        self.prob_grad   = float(np.clip(prob_grad, 0.0, 1.0))
        self.e1          = int(np.clip(e1, 1, 3))
        self.e2          = int(np.clip(e2, 1, 3))
        self.de_cr       = float(np.clip(de_cr, 0.0, 1.0))
        self.de_f        = float(max(0.0, de_f))
        self.sampling    = sampling

        # Tamanho de cada sub-arquivo (~1/3 do total)
        self.ns: int = max(1, self.pop_size // 3)

        # Tri-arquivo (inicializados em _initialize_advance)
        self.fa:  Population = Population.empty()
        self.da:  Population = Population.empty()
        self.fea: Population = Population.empty()

        # Contador de ID único por indivíduo (sem cache Broyden — EEJ é global)
        self._id_counter: int = 0

        # Ponto de referência ideal (mínimo por objetivo)
        self.zmin: Optional[np.ndarray] = None

        # EEJ — Jacobiano ensemble atual (atualizado a cada geração)
        # Forma: (n_obj, n_var) — partilhado por todos os offspring de gradiente
        self._eej: Optional[np.ndarray] = None

        self.nds = NonDominatedSorting()

    # ------------------------------------------------------------------
    # SETUP
    # ------------------------------------------------------------------

    def _setup(self, problem, **kwargs):
        """Configura vetores de referência e tamanho efetivo de sub-arquivo."""
        self.ns = int(max(1, self.pop_size // 3))

        assert problem is not None
        valid = (
            isinstance(self.ref_dirs, np.ndarray)
            and self.ref_dirs.ndim == 2
            and self.ref_dirs.shape[0] > 0
            and self.ref_dirs.shape[1] == int(problem.n_obj)
        )
        if not valid:
            W, n_eff  = UniformPoint(self.pop_size, int(problem.n_obj))
            self.ref_dirs = np.asarray(W, dtype=float)
            self.pop_size = int(max(4, n_eff))
            self.ns       = int(max(1, self.pop_size // 3))
        else:
            self.ref_dirs = np.asarray(self.ref_dirs, dtype=float)

    # ------------------------------------------------------------------
    # UTILITÁRIOS INTERNOS
    # ------------------------------------------------------------------

    def _next_id(self) -> int:
        """Gera próximo ID único de indivíduo."""
        self._id_counter += 1
        return self._id_counter

    def _update_eej(self) -> None:
        """
        Atualiza o Jacobiano Ensemble Evolutivo (EEJ) usando o arquivo FEA.

        Calcula J_hat = (dF^T @ dX) @ pinv(dX^T @ dX + reg·I) sobre todos
        os indivíduos não-dominados do FEA. Chamado uma vez por geração,
        ANTES de gerar offspring de gradiente.

        O EEJ estimado é armazenado em self._eej e compartilhado por todos
        os offspring de gradiente naquela geração.
        """
        if self.fea is None or len(self.fea) < 2:
            # Sem arquivo suficiente — mantém EEJ anterior (ou None)
            return

        X_fea = np.asarray(self.fea.get("X"), dtype=float)
        F_fea = _to_f(self.fea)

        self._eej = _compute_eej(X_fea, F_fea)

    def _prune_ids(self) -> None:
        """Mantém contador de IDs apenas; sem cache para limpar (EEJ é global)."""
        pass  # IDs ainda usados para padding/tracking nos arquivos

    # ------------------------------------------------------------------
    # CICLO DE VIDA EmoPyLab
    # ------------------------------------------------------------------

    def _initialize_infill(self) -> Population:
        """Gera população inicial aleatória."""
        pop = sample_initial(
            self.problem,
            self.pop_size,
            self.sampling,
            rng_from_algo(self),
        )
        for i in range(len(pop)):
            pop[i].set("id", self._next_id())
        return pop

    def _initialize_advance(self, infills=None, **kwargs) -> None:
        """Inicializa os três arquivos com a população avaliada."""
        if infills is None or len(infills) == 0:
            self.pop = Population.empty()
            self.fea = Population.empty()
            self.opt  = self.pop
            return

        # Atribui IDs aos indivíduos iniciais
        for i in range(len(infills)):
            if infills[i].get("id") is None:
                infills[i].set("id", self._next_id())

        # Ponto ideal
        F_init    = _to_f(infills)
        self.zmin = np.min(F_init, axis=0) - 1e-6

        # Inicializa arquivos
        self.fa  = Population.empty()
        self.da  = Population.empty()
        self.fea = infills
        self._update_archives(infills)

        # Calcula EEJ inicial sobre o FEA pós-inicialização
        self._update_eej()

        self.pop = self.fea
        self._set_optimum()

    def _infill(self) -> Population:
        """
        Gera offspring usando operador gradiente (prob_grad) ou DE/GA.
        As três fontes de variação usam os três arquivos: FA, DA, FEA.
        """
        rng = rng_from_algo(self)

        fea_safe = self.fea if self.fea is not None else Population.empty()
        if len(fea_safe) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, rng)

        pop1 = self.fa  if len(self.fa)  > 0 else self.fea
        pop2 = self.da  if len(self.da)  > 0 else self.fea
        pop3 = self.fea

        n3   = max(1, self.pop_size // 3)

        # Gera offspring dos três arquivos
        use_de = bool(rng.random() < 0.5)

        if use_de:
            off1 = self._operator_de(pop1, rng)
            off2 = self._operator_de(pop2, rng)
        else:
            off1 = self._operator_ga(pop1, rng)
            off2 = self._operator_ga(pop2, rng)

        # Terceiro bloco: gradient-based ou GA sobre FEA
        off3 = self._gradient_offspring(pop3, n3, rng)

        merged = Population.merge(off1, off2, off3)
        if len(merged) == 0:
            return sample_initial(self.problem, self.pop_size, self.sampling, rng)
        return merged

    def _advance(self, infills=None, **kwargs) -> None:
        """
        Avança o estado: atualiza EEJ, arquivos e população.
        """
        if infills is None or len(infills) == 0:
            return

        # Atribui IDs a offspring sem ID
        for i in range(len(infills)):
            if infills[i].get("id") is None:
                infills[i].set("id", self._next_id())

        # Atualiza os três arquivos
        self._update_archives(infills)
        self.pop = self.fea

        # Atualiza EEJ com o novo FEA (usado na PRÓXIMA geração)
        self._update_eej()

        self._set_optimum()

    # ------------------------------------------------------------------
    # OPERADORES DE VARIAÇÃO
    # ------------------------------------------------------------------

    def _gradient_offspring(
        self,
        pop: Population,
        n_out: int,
        rng: np.random.Generator,
    ) -> Population:
        """
        Gera n_out offspring via passo Euler-Maruyama estocástico:
            x_new = x - λ·q + ε·η·noise

        onde q é a direção de descida comum calculada pelo QP-simplex sobre o
        EEJ (Jacobiano Ensemble Evolutivo) do FEA atual.

        O EEJ é compartilhado por todos os offspring desta geração —
        calculado uma vez em _advance() sobre o arquivo FEA inteiro.
        """
        if len(pop) == 0:
            return Population.empty()

        X_pop  = np.asarray(pop.get("X"), dtype=float)
        cv_pop = _cv(pop)
        xl     = np.asarray(self.problem.xl, dtype=float)
        xu     = np.asarray(self.problem.xu, dtype=float)
        assert self.problem is not None
        n_var  = self.problem.n_var
        n_obj  = self.problem.n_obj

        # Seleção por torneio binário considerando CV
        cv_arr = np.asarray(
            TournamentSelection(2, n_out, cv_pop, rng=rng),
            dtype=int,
        ) - 1
        cv_arr = np.clip(cv_arr, 0, len(pop) - 1)

        # Usa o EEJ pré-calculado; fallback=zeros se ainda não disponível
        J_eej = self._eej if self._eej is not None else np.zeros((n_obj, n_var))

        # Calcular direção de descida comum UMA VEZ para toda a geração
        # (a mesma q é usada por todos os offspring de gradiente)
        q_eej = _compute_common_descent(J_eej)

        # Passo adaptativo: normaliza pelo range do espaço de decisão
        # λ_eff = step_size * mean(xu - xl) para escalar independente de n_var
        range_mean  = float(np.mean(xu - xl))
        lambda_eff  = self.step_size * range_mean

        offspring_X = np.empty((n_out, n_var), dtype=float)

        for k in range(n_out):
            p_idx = int(cv_arr[k])
            x     = X_pop[p_idx].copy()

            if rng.random() < self.prob_grad:
                # ---- Passo EEJ + Euler-Maruyama ----
                eta   = float(rng.random())
                noise = rng.standard_normal(n_var)
                x_new = x - lambda_eff * q_eej + self.noise_scale * eta * noise
            else:
                # ---- Mutação gaussiana centrada no pai (DE-like) ----
                scale = (xu - xl) * 0.1
                x_new = x + rng.standard_normal(n_var) * scale

            offspring_X[k] = np.clip(x_new, xl, xu)

        off = Population.new("X", offspring_X)
        for i in range(n_out):
            off[i].set("id", self._next_id())
        return off

    def _operator_de(self, pop: Population, rng: np.random.Generator) -> Population:
        """Aplica DE/rand/1/bin sobre o arquivo fornecido."""
        n = len(pop)
        if n == 0:
            return Population.empty()
        X   = np.asarray(pop.get("X"), dtype=float)
        mp1 = rng.permutation(n)
        mp2 = rng.permutation(n)
        try:
            off = OperatorDE(
                self.problem,
                X,
                X[mp1],
                X[mp2],
                Parameter=[self.de_cr, self.de_f, 1, 1],
                rng=rng,
            )
        except Exception:
            return Population.empty()
        off_X = np.asarray(off if not isinstance(off, Population) else off.get("X"), dtype=float)
        xl    = np.asarray(self.problem.xl, dtype=float)
        xu    = np.asarray(self.problem.xu, dtype=float)
        off_X = np.clip(off_X, xl, xu)
        result = Population.new("X", off_X)
        for i in range(len(result)):
            result[i].set("id", self._next_id())
        return result

    def _operator_ga(self, pop: Population, rng: np.random.Generator) -> Population:
        """Aplica SBX + PM sobre o arquivo fornecido."""
        n = len(pop)
        if n == 0:
            return Population.empty()
        X   = np.asarray(pop.get("X"), dtype=float)
        idx = rng.permutation(n)
        try:
            off = OperatorGA(
                self.problem,
                X[idx],
                Parameter=[1, 20, 1, 1],
                rng=rng,
            )
        except Exception:
            return Population.empty()
        off_X = np.asarray(off if not isinstance(off, Population) else off.get("X"), dtype=float)
        xl    = np.asarray(self.problem.xl, dtype=float)
        xu    = np.asarray(self.problem.xu, dtype=float)
        off_X = np.clip(off_X, xl, xu)
        result = Population.new("X", off_X)
        for i in range(len(result)):
            result[i].set("id", self._next_id())
        return result

    # ------------------------------------------------------------------
    # TRI-ARQUIVO
    # ------------------------------------------------------------------

    def _update_archives(self, offspring: Population) -> None:
        """Atualiza zmin e os três arquivos com o novo conjunto de offspring."""
        F_off  = _to_f(offspring)
        zmin_new = np.min(F_off, axis=0) - 1e-6

        if self.zmin is None:
            self.zmin = zmin_new
        else:
            self.zmin = np.minimum(self.zmin, zmin_new)

        self.fa  = self._forward_archive(self.fa,  offspring)
        self.da  = self._diversity_archive(self.da, offspring)
        self.fea = self._feasibility_archive(self.fea, offspring)

    def _forward_archive(
        self,
        fa: Population,
        offspring: Population,
    ) -> Population:
        """
        FA – Forward Exploration Archive.
        Mantém não-dominados (sem regra de restrição) com pressão de diversidade.
        """
        pool    = Population.merge(fa, offspring) if len(fa) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=False)
        pool    = pool[np.where(nd_mask)[0]]

        p = len(pool)
        if p == 0:
            return pool
        if p <= self.ns:
            if p < self.ns:
                # CORRIGIDO: usa rng controlado em vez de np.random.choice global
                rng_local = getattr(self, "random_state", None)
                if rng_local is None:
                    rng_local = np.random.default_rng()
                extra = self.ns - p
                idx_e = rng_local.choice(p, size=extra, replace=True)
                pool  = pool[np.concatenate([np.arange(p), idx_e])]
            return pool

        F       = _to_f(pool)
        zmax    = np.max(F, axis=0)
        zmin    = self.zmin if self.zmin is not None else (np.min(F, axis=0) - 1e-6)
        Fn      = _normalize_F(F, zmin, zmax)

        if self.e1 == 1:
            idx = _truncation_select(Fn, self.ns)
        elif self.e1 == 2:
            cd  = _crowding_distance(Fn)
            idx = np.argsort(-cd)[:self.ns]
        else:
            assert self.ref_dirs is not None
            idx = _nsga3_niching_select(
                Fn,
                np.asarray(self.ref_dirs, dtype=float),
                zmin,
                self.ns,
                rng=getattr(self, "random_state", None),
            )

        return pool[idx]

    def _feasibility_archive(
        self,
        fea: Population,
        offspring: Population,
    ) -> Population:
        """
        FEA – Feasibility Exploitation Archive.
        Mantém não-dominados com regra de restrição; prioriza viabilidade.
        """
        pool    = Population.merge(fea, offspring) if len(fea) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=True)
        pool    = pool[np.where(nd_mask)[0]]

        p = len(pool)
        if p == 0:
            return pool

        n_target = self.pop_size

        if p <= n_target:
            if p < n_target:
                # CORRIGIDO: usa rng controlado em vez de np.random.choice global
                rng_local = getattr(self, "random_state", None)
                if rng_local is None:
                    rng_local = np.random.default_rng()
                idx_e = rng_local.choice(p, size=n_target - p, replace=True)
                pool  = pool[np.concatenate([np.arange(p), idx_e])]
            return pool

        cv         = _cv(pool)
        n_feasible = int(np.sum(cv <= 0.0))

        if n_feasible <= n_target:
            # Não há viáveis suficientes: ordena por CV
            idx  = np.argsort(cv)[:n_target]
            return pool[idx]

        # Apenas viáveis
        pool_f = pool[np.where(cv <= 0.0)[0]]
        F      = _to_f(pool_f)
        zmin_f = np.min(F, axis=0) - 1e-6
        zmax_f = np.max(F, axis=0)
        Fn     = _normalize_F(F, zmin_f, zmax_f)

        if self.e2 == 1:
            idx = _truncation_select(Fn, n_target)
        elif self.e2 == 2:
            cd  = _crowding_distance(Fn)
            idx = np.argsort(-cd)[:n_target]
        else:
            assert self.ref_dirs is not None
            idx = _nsga3_niching_select(Fn, self.ref_dirs, zmin_f, n_target)

        return pool_f[idx]

    def _diversity_archive(
        self,
        da: Population,
        offspring: Population,
    ) -> Population:
        """
        DA – Diversity Enhancement Archive.
        Usa vetores de referência + ângulo seno para manter diversidade angular.
        Aplica regra de restrição no filtro de dominância.
        """
        pool    = Population.merge(da, offspring) if len(da) > 0 else offspring
        nd_mask = _constraint_pareto_nondominated(pool, add_constraint_rule=True)
        pool    = pool[np.where(nd_mask)[0]]

        if len(pool) == 0:
            return pool

        F  = _to_f(pool)
        cv = _cv(pool)
        n, m = F.shape

        W      = np.asarray(self.ref_dirs, dtype=float)
        ns_eff = len(W)

        # Threshold angular entre vetores de referência
        cos_ww  = _cosine_sim(W, W)
        ang_ww  = np.arccos(np.clip(cos_ww, -1.0, 1.0))
        np.fill_diagonal(ang_ww, np.inf)
        h = float(np.mean(np.min(ang_ww, axis=1)))

        # Normaliza objetivos
        feasible = cv <= 0.0
        if np.any(feasible):
            zmax = np.max(F[feasible], axis=0)
        else:
            zmax = F[int(np.argmin(cv))]

        zmin    = self.zmin if self.zmin is not None else (np.min(F, axis=0) - 1e-6)
        Fn      = _normalize_F(F, zmin, zmax)
        cos_ws  = _cosine_sim(W, Fn)
        sin_ws  = np.sqrt(np.maximum(1.0 - cos_ws * cos_ws, 0.0))

        chosen = np.empty(ns_eff, dtype=int)
        for i in range(ns_eff):
            angle = sin_ws[i]
            mask  = (angle <= h).copy()          # cópia writeable
            if not np.any(mask):
                mask[int(np.argmin(angle))] = True

            t = np.full(n, np.inf)
            t[mask] = cv[mask]

            feas_m = t <= 0.0
            if not np.any(feas_m):
                idx = int(np.argmin(t))
            else:
                t2 = np.full(n, np.inf)
                t2[feas_m] = angle[feas_m]
                idx = int(np.argmin(t2))
            chosen[i] = idx

        return pool[chosen]

    # ------------------------------------------------------------------
    # ÓTIMO
    # ------------------------------------------------------------------

    def _set_optimum(self) -> None:
        """Define self.opt como as soluções não-dominadas viáveis do FEA."""
        if self.pop is None or len(self.pop) == 0:
            self.opt = self.pop
            return

        cv          = _cv(self.pop)
        feasible_idx = np.where(cv <= 0.0)[0]

        if len(feasible_idx) > 0:
            feasible = self.pop[feasible_idx]
            nd       = self.nds.do(feasible.get("F"), only_non_dominated_front=True)
            self.opt = feasible[np.asarray(nd, dtype=int)]
        else:
            self.opt = self.pop[np.asarray([int(np.argmin(cv))], dtype=int)]
