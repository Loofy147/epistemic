import numpy as np
import epistem as ep

# 1. Feed written proposals
corpus = [
    ("status_quo",     "Maintain current single family zoning rules unchanged across the corridor, citing neighborhood stability and predictable property values for existing homeowners."),
    ("upzone",         "Rezone the full corridor for mid rise apartment buildings near the transit station, citing regional housing supply shortages and long commute times for essential workers."),
    ("transit_overlay","Allow mid rise apartments only within two blocks of the transit station, preserving existing single family zoning everywhere else in the corridor."),
    ("moratorium",     "Pause all zoning decisions along the corridor for eighteen months pending a comprehensive traffic and infrastructure capacity study."),
    ("density_bonus",  "Keep current zoning limits but offer density bonuses and expedited permitting to any developer who includes below market rate affordable units."),
    ("townhome_middle","Permit townhomes and small multiplexes corridor wide as a middle ground between single family and mid rise apartments, capped at three stories."),
]

# 2. Assign weights for each party
W = np.array([
    [0.40, 0.05, 0.20, 0.15, 0.20],   # Homeowners' priority share across 5 dims
    [0.05, 0.40, 0.20, 0.10, 0.25],   # Renters
    [0.05, 0.30, 0.15, 0.05, 0.45],   # Developers
])
party_names = ["Homeowners", "Renters", "Developers"]
dim_labels = ["density", "transit_access", "affordability", "process", "flexibility"]

print("--- 1 & 2: Embedding Texts and Resolving Consensus ---")
profiles = ep.embed(corpus, n_dims=5)
result = ep.lp_consensus(profiles, W, party_names=party_names)
print(result.summary("Zoning consensus"))
print()

print("--- 3: Scenario Stress-Testing ---")
stress_report = ep.stress(profiles, weight_matrix=W, n_scenarios=1000, seed=42)
print(stress_report.table(dim_labels))
print()

print("--- 4: Sensitivity Analysis ---")
# Check how far Homeowners (idx 0) would need to shift on "density" (idx 0) before recommendation flips
sens = ep.sensitivity(profiles, W, party_idx=0, dim_idx=0, deltas=[-0.15, -0.05, 0.0, 0.05, 0.15],
                     party_names=party_names, dim_labels=dim_labels)
print(sens.table())
print(f"Outcome flips? {sens.flips}")
print()

print("--- 5: Innovation Analysis ---")
# Bounded search beyond the convex hull around the current consensus v_opt
innov_res = ep.innovate(profiles, W, radius=0.08)
print(innov_res.summary())
