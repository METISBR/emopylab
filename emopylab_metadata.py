# EmoPyLab metadata definition for algorithm, problem, and metric discovery.

EMOPYLAB_ALGORITHMS = [
    {"class_name": "NSGA2", "name": "NSGA-II"},
    {"class_name": "NSGA3", "name": "NSGA-III"},
    {"class_name": "RVEA", "name": "RVEA"},
    {"class_name": "MOEAD", "name": "MOEA/D"},
    {"class_name": "AGE_MOEA", "name": "AGE-MOEA"},
    {"class_name": "AGE_MOEA2", "name": "AGE-MOEA-II"},
    {"class_name": "CTAEA", "name": "C-TAEA"},
    {"class_name": "SMSEMOA", "name": "SMS-EMOA"},
    {"class_name": "SPEA2", "name": "SPEA2"},
    {"class_name": "DNV_MaOEA", "name": "DNV-MaOEA"},
    {"class_name": "SSW_DNV", "name": "SSW-DNV"},
    {"class_name": "UNSGA3", "name": "U-NSGA-III"},
]

EMOPYLAB_PROBLEMS = {
    "dtlz": ["dtlz1", "dtlz2", "dtlz3", "dtlz4", "dtlz5", "dtlz6", "dtlz7"],
    "zdt": ["zdt1", "zdt2", "zdt3", "zdt4", "zdt5", "zdt6"],
    "wfg": ["wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]
}

EMOPYLAB_METRICS = {
    "HV": "Hypervolume",
    "IGD": "Inverted Generational Distance",
    "IGD+": "Inverted Generational Distance Plus",
    "GD": "Generational Distance",
    "GD+": "Generational Distance Plus",
    "Spacing": "Spacing",
    "R2": "R2 Indicator"
}

# Backward compatibility aliases
PYMOO_ALGORITHMS = EMOPYLAB_ALGORITHMS
PYMOO_PROBLEMS = EMOPYLAB_PROBLEMS
PYMOO_METRICS = EMOPYLAB_METRICS
