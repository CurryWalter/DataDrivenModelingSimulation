import time
import numpy as np
import matplotlib.pyplot as plt
import covasim as cv
import sciris as sc

N_RUNS = 30
N_DAYS = 365
POP_SIZE = 100000

base_pars = dict(
    pop_size=POP_SIZE,
    pop_infected=50,
    pop_type='hybrid',
    n_days=N_DAYS,
    verbose=0,
)

def testing(start):
        return cv.test_prob(symp_prob=0.2, asymp_prob=0.001,
                            symp_quar_prob=0.99, start_day=start, test_delay=1)

def tracing(start):
    return cv.contact_tracing(trace_probs=dict(h=1.0, s=0.5, w=0.5, c=0.2),
                                trace_time=dict(h=0, s=2, w=2, c=5),
                                start_day=start)


def get_scenarios():
    """Define scenarios"""
    scenario_1 = []
    scenario_2 = []
    scenario_3 = []
    scenario_4 = [testing(20), tracing(20)]
    scenario_5 = [testing(50), tracing(50)]
    
    return {
        'Scenario 1': scenario_1,
        'Scenario 2': scenario_2,
        'Scenario 3': scenario_3,
        'testing & tracing from day 20': scenario_4,
        'testing & tracing from day 50': scenario_5,
    }


def _extract_result(sim, key):
    res = sim.results[key]
    return np.asarray(res.values if hasattr(res, 'values') else res, dtype=float)


def run_single_replication(scen_name, seed, interventions):
    t0 = time.perf_counter()

    sim = cv.Sim(pars=base_pars, rand_seed=int(seed), interventions=interventions)
    sim.run()

    elapsed = time.perf_counter() - t0
    return {
        'scenario': scen_name,
        'seed': int(seed),
        'elapsed': elapsed,
        'new_infections': _extract_result(sim, 'new_infections'),
        'cum_infections': _extract_result(sim, 'cum_infections'),
        'n_severe': _extract_result(sim, 'n_severe'),
        'cum_deaths': _extract_result(sim, 'cum_deaths'),
    }


if __name__ == '__main__':
    scenarios = get_scenarios()

    # Generate a distinct, fresh random seed for every run on each execution
    raw_tasks = [
        (scen_name, intervs)
        for scen_name, intervs in scenarios.items()
        for _ in range(N_RUNS)
    ]
    fresh_seeds = np.random.SeedSequence().generate_state(len(raw_tasks))
    tasks = [
        (scen_name, int(seed), intervs)
        for (scen_name, intervs), seed in zip(raw_tasks, fresh_seeds)
    ]

    print(f"Starting {len(tasks)} simulations ({len(scenarios)} scenarios x {N_RUNS} runs) in parallel...")
    total_t0 = time.perf_counter()

    all_outputs = sc.parallelize(run_single_replication, iterarg=tasks)

    total_elapsed = time.perf_counter() - total_t0

    run_times = np.array([out['elapsed'] for out in all_outputs])
    print("\n=== Execution Time Summary ===")
    print(f"Total wall-clock time : {total_elapsed:.2f} s")
    print(f"Cumulative CPU time   : {run_times.sum():.2f} s (Effective speedup: {run_times.sum() / total_elapsed:.2f}x)")
    print(f"Per-run mean time     : {run_times.mean():.3f} s +/- {run_times.std():.3f} s")
    print(f"Per-run min / max     : {run_times.min():.3f} s / {run_times.max():.3f} s\n")

    for scen_name in scenarios:
        s_times = [o['elapsed'] for o in all_outputs if o['scenario'] == scen_name]
        print(f"  {scen_name:<35} | Mean run: {np.mean(s_times):.3f} s")

    metrics = [
        ('new_infections', 'Daily New Infections'),
        ('cum_infections', 'Cumulative Infections'),
        ('n_severe', 'Active Severe Cases (Hospital Demand)'),
        ('cum_deaths', 'Cumulative Deaths'),
    ]

    days = np.arange(N_DAYS + 1)
    colors = plt.cm.tab10.colors
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    axes = axes.flatten()

    for ax, (metric_key, title) in zip(axes, metrics):
        for idx, scen_name in enumerate(scenarios):
            runs_matrix = np.vstack([
                o[metric_key] for o in all_outputs if o['scenario'] == scen_name
            ])

            mean_traj = np.mean(runs_matrix, axis=0)
            ci_low = np.percentile(runs_matrix, 5, axis=0)
            ci_high = np.percentile(runs_matrix, 95, axis=0)

            color = colors[idx % len(colors)]
            ax.plot(days, mean_traj, label=scen_name, color=color, linewidth=2)
            ax.fill_between(days, ci_low, ci_high, color=color, alpha=0.18)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Agents')
        ax.grid(True, linestyle='--', alpha=0.4)

    axes[2].set_xlabel('Simulation Day')
    axes[3].set_xlabel('Simulation Day')
    axes[0].legend(loc='upper right', frameon=True, fontsize=9)

    fig.suptitle(
        f'Covasim What-If Policy Analysis (Mean & 90% CI across N={N_RUNS} runs/scenario)',
        fontsize=14,
        fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('covasim_whatif_results.png', dpi=300)
