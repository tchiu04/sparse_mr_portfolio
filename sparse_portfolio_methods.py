"""
Sparse Mean-Reverting Portfolio Optimization Methods

This module implements four different approaches for finding optimal sparse portfolios:
1. Exhaustive Search - tries all possible combinations (accurate but slow)
2. Greedy Forward Selection - builds portfolio incrementally (fast but suboptimal)
3. Simulated Annealing - heuristic global optimization (good balance)
4. Truncation Method - solves unconstrained then truncates (fast baseline)

Author: Implementation of "Sparse, mean reverting portfolio selection using simulated annealing"
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.linalg import eigh
from typing import Tuple, List, Set
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CORE MATHEMATICAL FUNCTIONS
# ============================================================================

def compute_covariance_matrices(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Γ₀ (contemporaneous) and Γ₁ (lag-1) covariance matrices
    
    Args:
        returns: T x n matrix of returns (T time periods, n assets)
    
    Returns:
        Gamma0: n x n contemporaneous covariance matrix
        Gamma1: n x n lag-1 cross-covariance matrix
    """
    # Remove any NaN values
    returns_clean = returns[~np.isnan(returns).any(axis=1)]
    
    # Γ₀ = E[x_t x_t^T] (contemporaneous covariance)
    Gamma0 = np.cov(returns_clean.T)
    
    # Γ₁ = E[x_t x_{t-1}^T] (lag-1 cross-covariance)
    x_t = returns_clean[1:]      # t = 1, 2, ..., T-1
    x_t_minus_1 = returns_clean[:-1]  # t = 0, 1, ..., T-2
    
    Gamma1 = np.cov(x_t.T, x_t_minus_1.T)[:len(x_t.T), len(x_t.T):]
    
    return Gamma0, Gamma1

def compute_VAR_matrices(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alternative interface for compatibility with paper notation
    Returns (Gamma0, A) where A is transition matrix from VAR(1): x_t = A x_{t-1} + ε_t
    """
    Gamma0, Gamma1 = compute_covariance_matrices(X)
    
    # Estimate A = Γ₁ Γ₀⁻¹ (VAR(1) transition matrix)
    try:
        A = Gamma1 @ np.linalg.pinv(Gamma0)
    except:
        A = Gamma1 @ np.linalg.pinv(Gamma0, rcond=1e-10)
    
    return Gamma0, A

def predictability_measure(weights: np.ndarray, Gamma0: np.ndarray, Gamma1: np.ndarray) -> float:
    """
    Compute predictability λ(w) = w^T Γ₁ w / w^T Γ₀ w
    Higher values indicate more mean-reverting (predictable) portfolios
    """
    numerator = weights.T @ Gamma1 @ weights
    denominator = weights.T @ Gamma0 @ weights
    
    # Avoid division by zero
    if abs(denominator) < 1e-12:
        return 0.0
    
    return numerator / denominator

def predictability_VAR(weights: np.ndarray, Gamma0: np.ndarray, A: np.ndarray) -> float:
    """
    Alternative predictability calculation using VAR formulation
    ν(w) = w^T A Γ₀ A^T w / w^T Γ₀ w
    """
    numerator = weights.T @ A @ Gamma0 @ A.T @ weights
    denominator = weights.T @ Gamma0 @ weights
    
    if abs(denominator) < 1e-12:
        return 0.0
        
    return numerator / denominator

# ============================================================================
# SPARSE PORTFOLIO CLASS
# ============================================================================

class SparsePortfolio:
    """Represents a sparse portfolio with L active assets out of n total assets"""
    
    def __init__(self, n_assets: int, L: int):
        self.n_assets = n_assets
        self.L = L  # Number of active assets
        self.active_assets = set()
        self.weights = np.zeros(n_assets)
        
    def initialize_random(self, seed: int = None):
        """Initialize with L random active assets and random weights"""
        if seed is not None:
            np.random.seed(seed)
            
        # Select L random assets
        self.active_assets = set(np.random.choice(self.n_assets, self.L, replace=False))
        
        # Assign random weights (can be positive or negative for long/short)
        self.weights = np.zeros(self.n_assets)
        for asset in self.active_assets:
            self.weights[asset] = np.random.randn()
            
        # Normalize weights to sum to 1
        if np.sum(np.abs(self.weights)) > 1e-12:
            self.weights = self.weights / np.sum(np.abs(self.weights))
    
    def initialize_greedy(self, X: np.ndarray, Gamma0: np.ndarray, Gamma1: np.ndarray):
        """Initialize using greedy selection for better starting point"""
        # Use greedy forward selection to get a good starting portfolio
        _, A = compute_VAR_matrices(X)
        selected = []
        
        # Greedy selection process
        for step in range(min(self.L, self.n_assets)):
            best_score = -np.inf
            best_asset = None
            best_weights = None
            
            candidates = [i for i in range(self.n_assets) if i not in selected]
            if not candidates:
                break
                
            for candidate in candidates:
                trial_indices = selected + [candidate]
                
                # Extract submatrices
                G0_S = Gamma0[np.ix_(trial_indices, trial_indices)]
                A_S = A[np.ix_(trial_indices, trial_indices)]
                
                try:
                    # Solve eigenvalue problem
                    vals, vecs = eigh(A_S @ G0_S @ A_S.T, G0_S)
                    top_w = vecs[:, np.argmax(vals)]
                    
                    if np.linalg.norm(top_w) > 1e-12:
                        top_w = top_w / np.linalg.norm(top_w)
                    
                    score = predictability_VAR(top_w, G0_S, A_S)
                    
                    if score > best_score:
                        best_score = score
                        best_asset = candidate
                        best_weights = top_w
                        
                except Exception:
                    continue
            
            if best_asset is not None:
                selected.append(best_asset)
            else:
                break
        
        # Set the portfolio to the greedy solution
        if len(selected) > 0 and best_weights is not None:
            self.active_assets = set(selected)
            self.weights = np.zeros(self.n_assets)
            for i, idx in enumerate(selected):
                self.weights[idx] = best_weights[i]
        else:
            # Fallback to random initialization
            self.initialize_random()
        
    def set_weights(self, active_indices: List[int], active_weights: np.ndarray):
        """Set portfolio weights for specific active assets"""
        self.active_assets = set(active_indices)
        self.weights = np.zeros(self.n_assets)
        for i, idx in enumerate(active_indices):
            self.weights[idx] = active_weights[i]
            
    def copy(self):
        """Create a deep copy of the portfolio"""
        new_portfolio = SparsePortfolio(self.n_assets, self.L)
        new_portfolio.active_assets = self.active_assets.copy()
        new_portfolio.weights = self.weights.copy()
        return new_portfolio
    
    def get_predictability(self, Gamma0: np.ndarray, Gamma1: np.ndarray) -> float:
        """Calculate predictability of current portfolio"""
        return predictability_measure(self.weights, Gamma0, Gamma1)

# ============================================================================
# METHOD 1: EXHAUSTIVE SEARCH
# ============================================================================

def exhaustive_search(X: np.ndarray, L: int, verbose: bool = True) -> Tuple[SparsePortfolio, float]:
    """
    🧮 Exhaustive Search: Try all (n choose L) subsets
    
    Pros: ✅ Guaranteed optimal solution
    Cons: ❌ Exponentially slow (use only for n ≤ 12)
    
    Args:
        X: T x n returns matrix
        L: Number of assets to select
        verbose: Print progress
    
    Returns:
        best_portfolio: Optimal sparse portfolio
        best_score: Best predictability score achieved
    """
    if verbose:
        print(f"🧮 EXHAUSTIVE SEARCH: Trying all combinations of {L} assets...")
    
    Gamma0, A = compute_VAR_matrices(X)
    n = X.shape[1]
    
    # Check feasibility
    total_combinations = np.math.comb(n, L)
    if total_combinations > 10000:
        raise ValueError(f"Too many combinations ({total_combinations:,})! Use n ≤ 12 for exhaustive search.")
    
    if verbose:
        print(f"Testing {total_combinations:,} combinations...")
    
    best_score = -np.inf
    best_portfolio = None
    
    for i, subset in enumerate(combinations(range(n), L)):
        S = list(subset)
        
        # Extract submatrices
        G0_S = Gamma0[np.ix_(S, S)]
        A_S = A[np.ix_(S, S)]
        
        # Solve generalized eigenvalue problem: A_S G0_S A_S^T v = λ G0_S v
        try:
            vals, vecs = eigh(A_S @ G0_S @ A_S.T, G0_S)
            top_w = vecs[:, np.argmax(vals)]
            
            # Normalize
            if np.linalg.norm(top_w) > 1e-12:
                top_w = top_w / np.linalg.norm(top_w)
            
            # Calculate predictability
            score = predictability_VAR(top_w, G0_S, A_S)
            
            if score > best_score:
                best_score = score
                best_portfolio = SparsePortfolio(n, L)
                best_portfolio.set_weights(S, top_w)
                
        except Exception as e:
            if verbose and i == 0:
                print(f"Warning: Numerical issue in subset {S}: {e}")
            continue
    
    if verbose:
        print(f"✅ Found optimal solution with predictability: {best_score:.6f}")
    
    return best_portfolio, best_score

# ============================================================================
# METHOD 2: GREEDY FORWARD SELECTION
# ============================================================================

def greedy_forward_selection(X: np.ndarray, L: int, verbose: bool = True) -> Tuple[SparsePortfolio, float]:
    """
    ⚙️ Greedy Forward Selection: Build portfolio one asset at a time
    
    Pros: ✅ Fast, intuitive
    Cons: ❌ Greedy, may miss global optimum
    
    Args:
        X: T x n returns matrix
        L: Number of assets to select
        verbose: Print progress
    
    Returns:
        best_portfolio: Greedy-optimal sparse portfolio
        final_score: Final predictability score
    """
    if verbose:
        print(f"⚙️ GREEDY SELECTION: Building portfolio incrementally...")
    
    Gamma0, A = compute_VAR_matrices(X)
    n = X.shape[1]
    selected = []
    
    for step in range(L):
        best_improvement = -np.inf
        best_asset = None
        best_weights = None
        
        # Try adding each remaining asset
        candidates = [i for i in range(n) if i not in selected]
        
        for candidate in candidates:
            trial_indices = selected + [candidate]
            
            # Extract submatrices
            G0_S = Gamma0[np.ix_(trial_indices, trial_indices)]
            A_S = A[np.ix_(trial_indices, trial_indices)]
            
            try:
                # Solve eigenvalue problem
                vals, vecs = eigh(A_S @ G0_S @ A_S.T, G0_S)
                top_w = vecs[:, np.argmax(vals)]
                
                if np.linalg.norm(top_w) > 1e-12:
                    top_w = top_w / np.linalg.norm(top_w)
                
                score = predictability_VAR(top_w, G0_S, A_S)
                
                if score > best_improvement:
                    best_improvement = score
                    best_asset = candidate
                    best_weights = top_w
                    
            except Exception:
                continue
        
        if best_asset is not None:
            selected.append(best_asset)
            if verbose:
                asset_name = f"Asset_{best_asset}"
                print(f"  Step {step+1}: Added {asset_name}, predictability = {best_improvement:.6f}")
        else:
            if verbose:
                print(f"  Step {step+1}: No improvement found, stopping early")
            break
    
    # Create final portfolio
    final_portfolio = SparsePortfolio(n, len(selected))
    if best_weights is not None and len(selected) > 0:
        final_portfolio.set_weights(selected, best_weights)
        final_score = best_improvement
    else:
        final_score = 0.0
    
    if verbose:
        print(f"✅ Greedy selection complete with {len(selected)} assets, predictability: {final_score:.6f}")
    
    return final_portfolio, final_score

# ============================================================================
# METHOD 3: SIMULATED ANNEALING (Enhanced)
# ============================================================================

def propose_new_portfolio(portfolio: SparsePortfolio, proposal_type: str = "random") -> SparsePortfolio:
    """Propose a new sparse portfolio by modifying the current one"""
    new_portfolio = portfolio.copy()

    if proposal_type == "random":
        proposal_type = np.random.choice(["swap", "perturb"], p=[0.5, 0.5])

    if proposal_type == "swap":
        active = list(new_portfolio.active_assets)
        inactive = [i for i in range(new_portfolio.n_assets) if i not in new_portfolio.active_assets]

        if active and inactive:
            remove = np.random.choice(active)
            add = np.random.choice(inactive)

            new_portfolio.active_assets.remove(remove)
            new_portfolio.weights[remove] = 0

            new_portfolio.active_assets.add(add)
            new_portfolio.weights[add] = np.random.normal(0, 0.1)

    elif proposal_type == "perturb":
        for asset in new_portfolio.active_assets:
            new_portfolio.weights[asset] += np.random.normal(0, 0.01)

    # Normalize
    norm = np.linalg.norm(new_portfolio.weights)
    if norm > 1e-8:
        new_portfolio.weights /= norm

    return new_portfolio

    
def simulated_annealing(X: np.ndarray, L: int, max_iterations: int = 5000,
                        initial_temperature: float = 1.0, cooling_rate: float = 0.8,
                        min_temperature: float = 1e-8, fallback_limit: int = 500,
                        verbose: bool = True) -> Tuple[SparsePortfolio, float, List[float]]:
    """
    🔥 Simulated Annealing: Heuristic global optimization
    
    Pros: ✅ Scales to large n, escapes local minima, good balance of exploration/exploitation
    Cons: ❌ Not deterministic, requires parameter tuning, may get stuck in local optima
    
    Args:
        X: T x n returns matrix
        L: Number of assets to select
        max_iterations: Number of optimization iterations
        initial_temperature: Starting temperature
        cooling_rate: Temperature decay rate
        min_temperature: Minimum temperature
        fallback_limit: Max iterations without improvement before fallback to best
        verbose: Print progress
    
    Returns:
        best_portfolio: Best sparse portfolio found
        best_score: Best predictability score
        score_history: History of scores during optimization
    """
    if verbose:
        print(f"🔥 SIMULATED ANNEALING: Enhanced optimization with greedy initialization...")
    
    Gamma0, Gamma1 = compute_covariance_matrices(X)
    n = X.shape[1]

    # Greedy initialization for using starting point from Greedy algorithm
    current_portfolio = SparsePortfolio(n, L)
    current_portfolio.initialize_greedy(X, Gamma0, Gamma1)
    current_score = current_portfolio.get_predictability(Gamma0, Gamma1)

    if verbose:
        print(f"   Initial greedy solution: λ = {current_score:.6f}")

    best_portfolio = current_portfolio.copy()
    best_score = current_score
    score_history = []

    temperature = initial_temperature
    fallback_counter = 0
    accepted_moves = 0

    for iteration in range(max_iterations):
        candidate = propose_new_portfolio(current_portfolio)
        candidate_score = candidate.get_predictability(Gamma0, Gamma1)

        delta = candidate_score - current_score

        # Acceptance criterion
        if delta > 0 or (temperature > min_temperature and np.random.rand() < np.exp(delta / temperature)):
            current_portfolio = candidate
            current_score = candidate_score
            fallback_counter = 0
            accepted_moves += 1

            if current_score > best_score:
                best_portfolio = current_portfolio.copy()
                best_score = current_score
        else:
            fallback_counter += 1

        # Cool down temperature every 100 iterations
        if iteration % 100 == 0:
            temperature = max(temperature * cooling_rate, min_temperature)

        # Fallback mechanism - return to best solution if stuck
        if fallback_counter >= fallback_limit:
            current_portfolio = best_portfolio.copy()
            current_score = best_score
            fallback_counter = 0

        score_history.append(current_score)

        # Progress reporting
        if verbose and (iteration + 1) % 1000 == 0:
            accept_rate = accepted_moves / (iteration + 1) * 100
            print(f"   Iter {iteration + 1:5d}: Best λ = {best_score:.6f}, "
                  f"Current λ = {current_score:.6f}, T = {temperature:.2e}, "
                  f"Accept = {accept_rate:.1f}%")

    if verbose:
        final_accept_rate = accepted_moves / max_iterations * 100
        improvement = (best_score - score_history[0]) / abs(score_history[0]) * 100
        print(f"✅ Optimization complete! Best λ = {best_score:.6f}")
        print(f"   Improvement over initial: {improvement:+.1f}%, Accept rate: {final_accept_rate:.1f}%")

    return best_portfolio, best_score, score_history


# ============================================================================
# METHOD 4: TRUNCATION
# ============================================================================

def truncation_method(X: np.ndarray, L: int, verbose: bool = True) -> Tuple[SparsePortfolio, float]:
    """
    ✂️ Truncation: Solve unconstrained, then keep top L weights
    
    Pros: ✅ Very fast, good baseline
    Cons: ❌ Ignores combinatorial structure, suboptimal
    
    Args:
        X: T x n returns matrix
        L: Number of assets to select
        verbose: Print progress
    
    Returns:
        sparse_portfolio: Truncated sparse portfolio
        score: Predictability score
    """
    if verbose:
        print(f"✂️ TRUNCATION METHOD: Solving unconstrained then keeping top {L} weights...")
    
    Gamma0, A = compute_VAR_matrices(X)
    n = X.shape[1]
    
    try:
        # Step 1: Solve full unconstrained generalized eigenvalue problem
        vals, vecs = eigh(A @ Gamma0 @ A.T, Gamma0)
        top_w = vecs[:, np.argmax(vals)]
        
        # Step 2: Select L largest absolute weights (paper's truncation step)
        abs_weights = np.abs(top_w)
        top_L_indices = np.argsort(abs_weights)[-L:]
        
        # Step 3: Re-solve in reduced L×L space (paper's key step!)
        G_prime = Gamma0[np.ix_(top_L_indices, top_L_indices)]
        AGA_prime = (A @ Gamma0 @ A.T)[np.ix_(top_L_indices, top_L_indices)]
        
        # Solve generalized eigenvalue problem in reduced space
        vals_reduced, vecs_reduced = eigh(AGA_prime, G_prime)
        optimal_weights_reduced = vecs_reduced[:, np.argmax(vals_reduced)]
        
        # Normalize reduced weights
        if np.linalg.norm(optimal_weights_reduced) > 1e-12:
            optimal_weights_reduced = optimal_weights_reduced / np.linalg.norm(optimal_weights_reduced)
        
        # Create sparse portfolio with re-optimized weights
        sparse_portfolio = SparsePortfolio(n, L)
        sparse_portfolio.set_weights(top_L_indices.tolist(), optimal_weights_reduced)
        
        # Calculate final score using re-optimized weights
        score = sparse_portfolio.get_predictability(Gamma0, A @ Gamma0 @ A.T)
        
        if verbose:
            print(f"✅ Truncation complete! Selected assets: {sorted(top_L_indices)}")
            print(f"   Predictability: {score:.6f}")
        
        return sparse_portfolio, score
        
    except Exception as e:
        if verbose:
            print(f"❌ Truncation failed: {e}")
        
        # Fallback: random selection
        fallback_portfolio = SparsePortfolio(n, L)
        fallback_portfolio.initialize_random()
        fallback_score = fallback_portfolio.get_predictability(Gamma0, A @ Gamma0 @ A.T)
        
        return fallback_portfolio, fallback_score

# ============================================================================
# COMPARISON FRAMEWORK
# ============================================================================

def compare_all_methods(X: np.ndarray, L: int, asset_names: List[str] = None, 
                       run_exhaustive: bool = None, verbose: bool = True) -> pd.DataFrame:
    """
    Compare all four optimization methods
    
    Args:
        X: T x n returns matrix
        L: Number of assets to select
        asset_names: Names of assets (optional)
        run_exhaustive: Whether to run exhaustive search (auto-decides if None)
        verbose: Print detailed results
    
    Returns:
        comparison_df: DataFrame with results from all methods
    """
    n = X.shape[1]
    if asset_names is None:
        asset_names = [f"Asset_{i}" for i in range(n)]
    
    # Decide whether to run exhaustive search
    if run_exhaustive is None:
        total_combinations = np.math.comb(n, L)
        run_exhaustive = total_combinations <= 5000
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"SPARSE PORTFOLIO OPTIMIZATION COMPARISON")
        print(f"{'='*60}")
        print(f"Assets: {n}, Sparsity: {L}, Combinations: {np.math.comb(n, L):,}")
        print(f"Running exhaustive search: {run_exhaustive}")
    
    results = {}
    
    # Method 1: Exhaustive Search
    if run_exhaustive:
        try:
            portfolio, score = exhaustive_search(X, L, verbose)
            results['Exhaustive'] = {
                'portfolio': portfolio,
                'score': score,
                'selected_assets': [asset_names[i] for i in sorted(portfolio.active_assets)],
                'weights': portfolio.weights[portfolio.weights != 0],
                'method': 'Exhaustive Search'
            }
        except Exception as e:
            if verbose:
                print(f"❌ Exhaustive search failed: {e}")
            results['Exhaustive'] = None
    else:
        if verbose:
            print("⏭️ Skipping exhaustive search (too many combinations)")
        results['Exhaustive'] = None
    
    # Method 2: Greedy Selection
    try:
        portfolio, score = greedy_forward_selection(X, L, verbose)
        results['Greedy'] = {
            'portfolio': portfolio,
            'score': score,
            'selected_assets': [asset_names[i] for i in sorted(portfolio.active_assets)],
            'weights': portfolio.weights[portfolio.weights != 0],
            'method': 'Greedy Selection'
        }
    except Exception as e:
        if verbose:
            print(f"❌ Greedy selection failed: {e}")
        results['Greedy'] = None
    
    # Method 3: Simulated Annealing
    try:
        portfolio, score, _ = simulated_annealing(X, L, verbose=verbose)
        results['SimAnnealing'] = {
            'portfolio': portfolio,
            'score': score,
            'selected_assets': [asset_names[i] for i in sorted(portfolio.active_assets)],
            'weights': portfolio.weights[portfolio.weights != 0],
            'method': 'Simulated Annealing'
        }
    except Exception as e:
        if verbose:
            print(f"❌ Simulated annealing failed: {e}")
        results['SimAnnealing'] = None
    
    # Method 4: Truncation
    try:
        portfolio, score = truncation_method(X, L, verbose)
        results['Truncation'] = {
            'portfolio': portfolio,
            'score': score,
            'selected_assets': [asset_names[i] for i in sorted(portfolio.active_assets)],
            'weights': portfolio.weights[portfolio.weights != 0],
            'method': 'Truncation'
        }
    except Exception as e:
        if verbose:
            print(f"❌ Truncation method failed: {e}")
        results['Truncation'] = None
    
    # Create comparison DataFrame
    comparison_data = []
    for method_name, result in results.items():
        if result is not None:
            comparison_data.append({
                'Method': result['method'],
                'Predictability': result['score'],
                'Selected_Assets': ', '.join(result['selected_assets']),
                'Num_Assets': len(result['selected_assets'])
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    if len(comparison_df) > 0:
        comparison_df = comparison_df.sort_values('Predictability', ascending=False)
        
        if verbose:
            print(f"\n{'='*60}")
            print("FINAL COMPARISON")
            print(f"{'='*60}")
            print(comparison_df.to_string(index=False, float_format='%.6f'))
            
            best_method = comparison_df.iloc[0]['Method']
            best_score = comparison_df.iloc[0]['Predictability']
            print(f"\n🏆 Winner: {best_method} (λ = {best_score:.6f})")
    
    return comparison_df, results