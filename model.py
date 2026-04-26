import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field
from typing import Optional

RNG = np.random.default_rng(42)
N_SAMPLES = 20000
PROJECT_TO_YEAR = 2070
CURRENT_YEAR = 2026


# =============================================================================
# DATA — CURATED
# Each entry is (year, USABLE_qubits_at_stated_fidelity, stated_2q_error).
# "Usable" means the largest contiguous subset of the device that achieves the
# error rate in published benchmarks (NOT total fabricated qubits on the die).
#
# Curation choices, documented:
#   - IBM Condor (1121 q, 2023) REMOVED: never used for computation,
#     no published benchmark at full size.
#   - IBM Kookaburra projection REMOVED: not yet delivered as of 2025.
#   - We use annual maxima per modality to avoid double-counting refreshes.
#   - Verified against published 2Q gate fidelities, not marketing slides.
# =============================================================================

# Superconducting — annual maxima of "usable qubits at stated fidelity"
SUPERCONDUCTING = [
    (2017,  16, 5.0e-2),    # IBM 16-qubit
    (2018,  20, 3.0e-2),    # IBM Q20
    (2019,  53, 1.5e-2),    # Google Sycamore (supremacy run)
    (2020,  65, 1.0e-2),    # IBM Hummingbird
    (2021, 127, 8.0e-3),    # IBM Eagle (full-chip benchmarks)
    (2022, 127, 7.0e-3),    # Eagle r3 improved
    (2023, 133, 5.0e-3),    # IBM Heron r1
    (2024, 105, 3.0e-3),    # Google Willow (full-chip QEC demo)
    (2025, 156, 3.0e-3),    # IBM Heron r2 (verified)
]

# Trapped ion — annual maxima
TRAPPED_ION = [
    (2018,  11, 5.0e-3),
    (2019,  20, 4.0e-3),
    (2020,  32, 3.0e-3),
    (2021,  20, 2.0e-3),    # Quantinuum H1 (full-chip)
    (2022,  32, 1.5e-3),    # Quantinuum H2
    (2023,  32, 8.0e-4),    # H2 improved
    (2024,  56, 1.0e-3),    # H2-1 expansion (verified)
    (2025,  56, 8.0e-4),    # H2-1 improved
]

# Mosca / GRI expert survey (unchanged)
MOSCA_SURVEY = [
    (2030, 40,  6),
    (2035, 40, 18),
    (2040, 40, 28),
    (2045, 40, 32),
    (2050, 40, 35),
]


# =============================================================================
# PHYSICAL FLOORS AND CEILINGS PER MODALITY
# =============================================================================

@dataclass
class ModalityLimits:
    name: str
    error_floor: float          # asymptotic best 2Q error rate
    qubit_ceiling_log: float    # asymptotic monolithic qubit ceiling (log_e)
    ceiling_softness: float     # how gradually the ceiling kicks in

SUPERCONDUCTING_LIMITS = ModalityLimits(
    name="superconducting",
    error_floor=1e-5,            # Materials / TLS limited; debatable
    qubit_ceiling_log=np.log(1e7),  # ~10M monolithic; modular can exceed
    ceiling_softness=2.0,
)

TRAPPED_ION_LIMITS = ModalityLimits(
    name="trapped_ion",
    error_floor=1e-6,            # Motional heating, laser noise limited
    qubit_ceiling_log=np.log(1e5),  # Ion trap scaling is harder; modular helps
    ceiling_softness=2.0,
)


def apply_qubit_ceiling(log_qubits, limits: ModalityLimits):
    """Soft cap on log qubit count using a logistic-like saturation."""
    L = limits.qubit_ceiling_log
    s = limits.ceiling_softness
    # Smoothly saturate at L: y = L - s * log(1 + exp((L - x) / s))
    return L - s * np.log1p(np.exp((L - log_qubits) / s))


def apply_error_floor(error_rate, limits: ModalityLimits):
    """Soft floor on error rate (asymptote at error_floor)."""
    floor = limits.error_floor
    # Smooth max: error stays above floor
    return floor + np.maximum(error_rate - floor, 0) * (
        error_rate / (error_rate + floor)
    )


# =============================================================================
# RESOURCE ESTIMATE
# =============================================================================

@dataclass
class ResourceEstimate:
    name: str
    logical_qubits: int
    toffoli_count: float
    target_logical_error: float
    runtime_budget_hours: float

SECP256K1 = ResourceEstimate(
    name="secp256k1 ECDSA",
    logical_qubits=2330,
    toffoli_count=1.3e11,
    target_logical_error=1e-12,
    runtime_budget_hours=24.0,
)

# Surface code parameters
SURFACE_CODE_THRESHOLD = 1e-2
LOGICAL_ERROR_PREFACTOR = 0.1


# =============================================================================
# CONJUGATE BAYESIAN LINEAR REGRESSION (same as before)
# =============================================================================

def fit_conjugate_linear(years, log_values):
    years = np.asarray(years, dtype=float)
    y = np.asarray(log_values, dtype=float)
    year0 = years.mean()
    x = years - year0
    n = len(y)
    X = np.column_stack([np.ones(n), x])

    Lambda_0 = np.diag([0.01, 0.01])
    mu_0 = np.array([y.mean(), 0.0])
    alpha_0 = 1.0
    beta_0 = 1.0

    Lambda_n = X.T @ X + Lambda_0
    mu_n = np.linalg.solve(Lambda_n, X.T @ y + Lambda_0 @ mu_0)
    alpha_n = alpha_0 + n / 2
    beta_n = beta_0 + 0.5 * (
        y @ y + mu_0 @ Lambda_0 @ mu_0 - mu_n @ Lambda_n @ mu_n
    )

    return {
        "year0": year0, "mu_n": mu_n, "Lambda_n": Lambda_n,
        "alpha_n": alpha_n, "beta_n": beta_n, "n": n,
    }


def sample_posterior(post, n_samples, rng):
    sigma2 = stats.invgamma.rvs(
        a=post["alpha_n"], scale=post["beta_n"],
        size=n_samples, random_state=rng,
    )
    cov_factor = np.linalg.inv(post["Lambda_n"])
    L = np.linalg.cholesky(cov_factor)
    samples = np.zeros((n_samples, 2))
    z = rng.standard_normal((n_samples, 2))
    for i in range(n_samples):
        samples[i] = post["mu_n"] + np.sqrt(sigma2[i]) * (L @ z[i])
    return samples[:, 0], samples[:, 1], sigma2


def project_log(intercepts, slopes, sigmas2, year, year0,
                include_noise=True, rng=None):
    mean = intercepts + slopes * (year - year0)
    if include_noise:
        if rng is None:
            rng = np.random.default_rng()
        noise = rng.standard_normal(len(intercepts)) * np.sqrt(sigmas2)
        return mean + noise
    return mean


def physical_per_logical(p_phys, target_logical_error,
                         p_th=SURFACE_CODE_THRESHOLD,
                         A=LOGICAL_ERROR_PREFACTOR):
    p_phys = np.asarray(p_phys)
    target_logical_error = np.asarray(target_logical_error)
    safe = p_phys < p_th
    ratio = np.where(safe, p_th / np.maximum(p_phys, 1e-30), 1.0)
    log_ratio = np.log(ratio)
    log_A_over_PL = np.log(A / target_logical_error)
    d_real = 2 * log_A_over_PL / np.maximum(log_ratio, 1e-9) - 1
    d = np.ceil(np.maximum(d_real, 3))
    physical = 2 * d * d
    return np.where(safe, physical, np.inf)


# =============================================================================
# Q-DAY MONTE CARLO — HARDENED
# Now with: physical floors, qubit ceilings, scenario modifiers.
# =============================================================================

@dataclass
class Scenario:
    name: str
    # Multiplicative slowdown applied to slope after slowdown_year
    slowdown_factor: float = 1.0
    slowdown_year: int = 2027
    # One-time algorithmic improvement (multiplicative on logical_qubits needed)
    breakthrough_factor: float = 1.0    # 1.0 = no breakthrough
    breakthrough_prob: float = 0.0      # per-sample probability it happens
    breakthrough_year_range: tuple = (2027, 2045)
    # Paradigm shift: independent modality reaches CRQC at this year
    paradigm_shift_year: Optional[float] = None
    paradigm_shift_year_sd: float = 5.0
    paradigm_shift_prob: float = 0.0    # per-sample probability

SCENARIOS = {
    "status_quo": Scenario(
        name="Status quo (trends continue, with floors)",
    ),
    "stagnation": Scenario(
        name="Stagnation (post-2027 progress at 50%)",
        slowdown_factor=0.5,
        slowdown_year=2027,
    ),
    "breakthrough": Scenario(
        name="Algorithmic breakthrough (10x reduction, 30% chance)",
        breakthrough_factor=0.1,
        breakthrough_prob=0.30,
    ),
    "paradigm_shift": Scenario(
        name="New modality wins (15% chance, ~2042)",
        paradigm_shift_year=2042,
        paradigm_shift_year_sd=5.0,
        paradigm_shift_prob=0.15,
    ),
}


def run_qday_hardened(qubit_post, error_post, limits: ModalityLimits,
                      resource: ResourceEstimate, scenario: Scenario,
                      n_samples=N_SAMPLES, rng=RNG,
                      end_year=PROJECT_TO_YEAR, start_year=CURRENT_YEAR):
    q_int, q_slo, q_sig2 = sample_posterior(qubit_post, n_samples, rng)
    e_int, e_slo, e_sig2 = sample_posterior(error_post, n_samples, rng)

    # Algorithmic gradual improvement
    algo_rate = rng.normal(0.05, 0.03, size=n_samples)
    algo_rate = np.clip(algo_rate, 0.0, 0.15)

    # One-time breakthrough events
    has_breakthrough = rng.uniform(size=n_samples) < scenario.breakthrough_prob
    breakthrough_year = rng.integers(
        scenario.breakthrough_year_range[0],
        scenario.breakthrough_year_range[1] + 1,
        size=n_samples,
    )

    # Paradigm shift events
    has_paradigm = rng.uniform(size=n_samples) < scenario.paradigm_shift_prob
    if scenario.paradigm_shift_year is not None:
        paradigm_year = rng.normal(
            scenario.paradigm_shift_year,
            scenario.paradigm_shift_year_sd,
            size=n_samples,
        ).astype(int)
    else:
        paradigm_year = np.full(n_samples, end_year + 100)

    qday = np.full(n_samples, end_year + 1, dtype=int)
    base_logical_needed = resource.logical_qubits

    for year in range(start_year, end_year + 1):
        # Apply slowdown to slopes if past slowdown_year
        if year > scenario.slowdown_year:
            years_post_slow = year - scenario.slowdown_year
            effective_q_slope = q_slo * scenario.slowdown_factor
            effective_e_slope = e_slo * scenario.slowdown_factor
            # Compute as if slope changed at slowdown_year
            log_qubits_at_slow = q_int + q_slo * (
                scenario.slowdown_year - qubit_post["year0"]
            )
            log_inv_err_at_slow = e_int + e_slo * (
                scenario.slowdown_year - error_post["year0"]
            )
            log_qubits = log_qubits_at_slow + effective_q_slope * years_post_slow
            log_inv_err = log_inv_err_at_slow + effective_e_slope * years_post_slow
        else:
            log_qubits = project_log(q_int, q_slo, q_sig2, year,
                                     qubit_post["year0"], rng=rng)
            log_inv_err = project_log(e_int, e_slo, e_sig2, year,
                                      error_post["year0"], rng=rng)

        # Add small noise
        noise_q = rng.standard_normal(n_samples) * np.sqrt(q_sig2) * 0.5
        noise_e = rng.standard_normal(n_samples) * np.sqrt(e_sig2) * 0.5
        log_qubits = log_qubits + noise_q
        log_inv_err = log_inv_err + noise_e

        # Apply ceilings and floors
        log_qubits_capped = apply_qubit_ceiling(log_qubits, limits)
        physical_qubits = np.exp(log_qubits_capped)

        error_rate_raw = 1.0 / np.exp(log_inv_err)
        error_rate = apply_error_floor(error_rate_raw, limits)

        # Required logical qubits (gradual algo improvement)
        years_elapsed = year - start_year
        logical_needed = base_logical_needed * (1 - algo_rate) ** years_elapsed
        # Apply breakthrough if it has occurred by now
        breakthrough_active = has_breakthrough & (year >= breakthrough_year)
        logical_needed = np.where(
            breakthrough_active,
            logical_needed * scenario.breakthrough_factor,
            logical_needed,
        )
        logical_needed = np.maximum(logical_needed, 50)

        # Capability
        per_logical = physical_per_logical(error_rate, resource.target_logical_error)
        logical_available = physical_qubits / per_logical

        # Crossing
        hardware_crosses = (logical_available >= logical_needed) & (qday > end_year)
        # Paradigm shift triggers independently
        paradigm_crosses = has_paradigm & (year >= paradigm_year) & (qday > end_year)
        crosses = hardware_crosses | paradigm_crosses
        qday = np.where(crosses, year, qday)

    return qday


# =============================================================================
# MOSCA EXPERT ELICITATION — with widened uncertainty
# We treat the survey as 5 anchor points, but interpolate with added noise
# reflecting that experts disagree more than 5 anchors capture.
# =============================================================================

def expert_implied_qday_samples(survey, n_samples=N_SAMPLES, rng=RNG):
    posts = {}
    for year, n_exp, k in survey:
        a, b = k + 1, n_exp - k + 1
        posts[year] = stats.beta.rvs(a, b, size=n_samples, random_state=rng)
    years = sorted(posts.keys())
    cdf_matrix = np.column_stack([posts[y] for y in years])
    cdf_matrix = np.maximum.accumulate(cdf_matrix, axis=1)
    cdf_matrix = np.clip(cdf_matrix, 0, 1)

    # Inverse-CDF with linear interpolation between anchors
    u = rng.uniform(size=n_samples)
    qday = np.full(n_samples, years[-1] + 15, dtype=float)
    years_arr = np.array(years, dtype=float)
    for i in range(n_samples):
        cdf = cdf_matrix[i]
        # Prepend (start_year, 0) and (last_year + 20, 1) for interpolation
        xs = np.concatenate([[CURRENT_YEAR], years_arr, [years_arr[-1] + 20]])
        ys = np.concatenate([[0.0], cdf, [1.0]])
        # Ensure monotonic
        ys = np.maximum.accumulate(ys)
        if u[i] >= ys[-1]:
            qday[i] = xs[-1]
        else:
            qday[i] = np.interp(u[i], ys, xs)
    return qday


# =============================================================================
# SUMMARY, MOSCA INEQUALITY, PLOTTING
# =============================================================================

def summarize(name, qday_samples):
    finite = qday_samples[qday_samples <= PROJECT_TO_YEAR]
    pct = 100 * len(finite) / len(qday_samples)
    print(f"\n--- {name} ---")
    print(f"P(Q-day by {PROJECT_TO_YEAR}) = {pct:.1f}%")
    if len(finite) > 0:
        for q in [2.5, 25, 50, 75, 97.5]:
            v = np.percentile(finite, q)
            print(f"  {q:5.1f}th percentile: {v:.0f}")
        for milestone in [2030, 2035, 2040, 2045, 2050, 2060]:
            p = 100 * np.mean(qday_samples <= milestone)
            print(f"  P(Q-day <= {milestone}) = {p:5.1f}%")


def mosca_threat_probability(qday_samples, x_years,
                             y_mean=10, y_sd=3, rng=RNG):
    z_years = qday_samples - CURRENT_YEAR
    y_samples = np.maximum(rng.normal(y_mean, y_sd, len(qday_samples)), 1.0)
    return float(np.mean(x_years + y_samples > z_years))


def plot_results(scenario_results, expert_qday, qubit_posts, error_posts,
                 limits_dict, resource, fname):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top-left: Q-day CDF across scenarios
    ax = axes[0, 0]
    years_grid = np.arange(CURRENT_YEAR, PROJECT_TO_YEAR + 1)
    colors = {"status_quo": "tab:blue", "stagnation": "tab:gray",
              "breakthrough": "tab:red", "paradigm_shift": "tab:purple"}
    for sc_key, samples in scenario_results.items():
        cdf = [np.mean(samples <= y) for y in years_grid]
        ax.plot(years_grid, cdf, label=SCENARIOS[sc_key].name,
                color=colors[sc_key], linewidth=2)
    cdf_exp = [np.mean(expert_qday <= y) for y in years_grid]
    ax.plot(years_grid, cdf_exp, label="Expert elicitation",
            color="tab:green", linewidth=2, linestyle="--")
    ax.set_xlabel("Year")
    ax.set_ylabel("P(Q-day ≤ year)")
    ax.set_title(f"Q-day CDF by scenario — {resource.name}")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 1)

    # Top-right: Q-day distributions
    ax = axes[0, 1]
    bins = np.arange(CURRENT_YEAR, PROJECT_TO_YEAR + 2, 2)
    for sc_key, samples in scenario_results.items():
        finite = samples[samples <= PROJECT_TO_YEAR]
        ax.hist(finite, bins=bins, alpha=0.4,
                label=sc_key, color=colors[sc_key])
    ax.set_xlabel("Q-day year")
    ax.set_ylabel("Posterior samples")
    ax.set_title("Q-day distribution by scenario")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Bottom-left: Qubit projections with ceilings
    ax = axes[1, 0]
    for post, raw, label, color, lim in [
        (qubit_posts["sc"], SUPERCONDUCTING, "Superconducting", "tab:blue",
         limits_dict["sc"]),
        (qubit_posts["ti"], TRAPPED_ION, "Trapped ion", "tab:orange",
         limits_dict["ti"]),
    ]:
        years_data = [r[0] for r in raw]
        qubits_data = [r[1] for r in raw]
        ax.scatter(years_data, qubits_data, color=color, s=30, zorder=3)

        proj_years = np.arange(min(years_data), 2055)
        intercepts, slopes, sigmas2 = sample_posterior(
            post, 2000, np.random.default_rng(0))
        log_proj = np.array([
            apply_qubit_ceiling(
                project_log(intercepts, slopes, sigmas2, y, post["year0"],
                            rng=np.random.default_rng(y)),
                lim,
            )
            for y in proj_years
        ])
        proj = np.exp(log_proj)
        med = np.median(proj, axis=1)
        lo = np.percentile(proj, 2.5, axis=1)
        hi = np.percentile(proj, 97.5, axis=1)
        ax.plot(proj_years, med, color=color, label=f"{label} (median)")
        ax.fill_between(proj_years, lo, hi, color=color, alpha=0.15)
        ax.axhline(np.exp(lim.qubit_ceiling_log), color=color,
                   linestyle=":", alpha=0.5)

    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("Usable physical qubits")
    ax.set_title("Qubit projection (with soft ceilings)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    # Bottom-right: Error rate projections with floors
    ax = axes[1, 1]
    for post, raw, label, color, lim in [
        (error_posts["sc"], SUPERCONDUCTING, "Superconducting", "tab:blue",
         limits_dict["sc"]),
        (error_posts["ti"], TRAPPED_ION, "Trapped ion", "tab:orange",
         limits_dict["ti"]),
    ]:
        years_data = [r[0] for r in raw]
        err_data = [r[2] for r in raw]
        ax.scatter(years_data, err_data, color=color, s=30, zorder=3)

        proj_years = np.arange(min(years_data), 2055)
        intercepts, slopes, sigmas2 = sample_posterior(
            post, 2000, np.random.default_rng(1))
        log_inv_proj = np.array([
            project_log(intercepts, slopes, sigmas2, y, post["year0"],
                        rng=np.random.default_rng(y + 100))
            for y in proj_years
        ])
        err_proj_raw = 1.0 / np.exp(log_inv_proj)
        err_proj = np.array([apply_error_floor(e, lim) for e in err_proj_raw])
        med = np.median(err_proj, axis=1)
        lo = np.percentile(err_proj, 2.5, axis=1)
        hi = np.percentile(err_proj, 97.5, axis=1)
        ax.plot(proj_years, med, color=color, label=f"{label} (median)")
        ax.fill_between(proj_years, lo, hi, color=color, alpha=0.15)
        ax.axhline(lim.error_floor, color=color, linestyle=":", alpha=0.5)

    ax.axhline(SURFACE_CODE_THRESHOLD, color="red", linestyle="--",
               label=f"SC threshold ({SURFACE_CODE_THRESHOLD:.0e})")
    ax.set_yscale("log")
    ax.set_xlabel("Year")
    ax.set_ylabel("2Q gate error rate")
    ax.set_title("Error rate projection (with soft floors)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("HARDENED Q-DAY MODEL")
    print("Curated dataset + physical floors + scenario analysis")
    print("=" * 70)

    sc_years = [r[0] for r in SUPERCONDUCTING]
    sc_log_q = [np.log(r[1]) for r in SUPERCONDUCTING]
    sc_log_inv_e = [np.log(1.0 / r[2]) for r in SUPERCONDUCTING]

    ti_years = [r[0] for r in TRAPPED_ION]
    ti_log_q = [np.log(r[1]) for r in TRAPPED_ION]
    ti_log_inv_e = [np.log(1.0 / r[2]) for r in TRAPPED_ION]

    qubit_posts = {
        "sc": fit_conjugate_linear(sc_years, sc_log_q),
        "ti": fit_conjugate_linear(ti_years, ti_log_q),
    }
    error_posts = {
        "sc": fit_conjugate_linear(sc_years, sc_log_inv_e),
        "ti": fit_conjugate_linear(ti_years, ti_log_inv_e),
    }
    limits_dict = {"sc": SUPERCONDUCTING_LIMITS, "ti": TRAPPED_ION_LIMITS}

    print("\nCurated trends (slope = log-units / year):")
    for label, post in [
        ("SC log(usable qubits)", qubit_posts["sc"]),
        ("TI log(usable qubits)", qubit_posts["ti"]),
        ("SC log(1/error)", error_posts["sc"]),
        ("TI log(1/error)", error_posts["ti"]),
    ]:
        b = post["mu_n"][1]
        print(f"  {label:28s}  slope = {b:+.3f}  "
              f"({np.exp(b * 10):.1f}× per decade)")

    # For each scenario, compute Q-day from BEST modality (min crossing)
    # We sample from both modalities and take per-sample minimum.
    print("\n" + "=" * 70)
    print("Q-day = first crossing across BOTH modalities (best of)")
    print("=" * 70)

    scenario_results = {}
    for sc_key, scenario in SCENARIOS.items():
        rng_local = np.random.default_rng(123)
        qday_sc = run_qday_hardened(
            qubit_posts["sc"], error_posts["sc"], SUPERCONDUCTING_LIMITS,
            SECP256K1, scenario, rng=rng_local,
        )
        rng_local = np.random.default_rng(124)
        qday_ti = run_qday_hardened(
            qubit_posts["ti"], error_posts["ti"], TRAPPED_ION_LIMITS,
            SECP256K1, scenario, rng=rng_local,
        )
        # Best-of: a CRQC counts if any modality reaches it
        scenario_results[sc_key] = np.minimum(qday_sc, qday_ti)
        summarize(scenario.name, scenario_results[sc_key])

    expert_qday = expert_implied_qday_samples(MOSCA_SURVEY)
    summarize("Expert elicitation (Mosca/GRI)", expert_qday)

    # Mosca inequality across scenarios
    print("\n" + "=" * 70)
    print("MOSCA INEQUALITY for Bitcoin")
    print("Y (migration) ~ Normal(10, 3) years")
    print("=" * 70)
    print(f"\n{'Scenario':<45} " + "  ".join(
        f"X={x:2d}" for x in [5, 10, 15, 20, 30]))
    for sc_key, samples in scenario_results.items():
        row = [SCENARIOS[sc_key].name[:43]]
        for x in [5, 10, 15, 20, 30]:
            p = mosca_threat_probability(samples, x_years=x)
            row.append(f"{100*p:4.0f}%")
        print(f"{row[0]:<45} " + "  ".join(row[1:]))

    plot_results(scenario_results, expert_qday, qubit_posts, error_posts,
                 limits_dict, SECP256K1,
                 "/home/claude/qday/qday_hardened.png")
    print("\nPlot saved to qday_hardened.png")


if __name__ == "__main__":
    main()
