from typing import List
import numpy as np

__all__ = [
    "Card",
    "generate_domain_matrix",
    "assign_int_to_matrix",
    "assign_symbol_to_matrix",
    "generate_card_list",
    "generate_card_list_from_domain_dict"
]

RANDOM_SEED = None # # no seed = non-reproducible

class Card:
    def __init__(self, *symbols):
        self.symbols = symbols
    def __repr__(self) -> str:
        return f"({' '.join([str(sym) for sym in self.symbols ])})"

def gauss_2d(x, y, center_x=0, center_y=0, sigma_x=1, sigma_y=1):
    return 1. / (2. * np.pi * sigma_x * sigma_y) * np.exp(-((x - center_x)**2. / (2. * sigma_x**2.) + (y - center_y)**2. / (2. * sigma_y**2.)))

def generate_domain_matrix(matrix_side: int, n_domains: int) -> np.ndarray:
    
    rdn = np.random.default_rng(seed=RANDOM_SEED)

    domains = np.arange(n_domains, dtype=int)
    
    # Spread the domain values among the map
    domain_map = np.empty((matrix_side, matrix_side), dtype=int)

    # Find the center of each domain island: in a circle around the center of the grid
    angles = np.radians( np.arange(n_domains) * int(360 / n_domains) )
    matrix_center = matrix_side / 2
    radius = matrix_side / 4
    sigma = matrix_side / 5
    domain_centers_x = (matrix_center + radius * np.cos(angles)).astype(int)
    domain_centers_y = (matrix_center + radius * np.sin(angles)).astype(int)
    
    for i in range(matrix_side):
        for j in range(matrix_side):
            domain_probability = gauss_2d(i, j, center_x=domain_centers_x, center_y=domain_centers_y, sigma_x=sigma, sigma_y=sigma)
            domain_map[i, j] = rdn.choice(domains, size=1, p=domain_probability/domain_probability.sum())[0]

    return domain_map

def assign_int_to_matrix(matrix: np.ndarray, n_symbols: int) -> np.ndarray:
    rdn = np.random.default_rng(seed=RANDOM_SEED)

    # Assign a symbol for each domain, uniformely distributed among n_symbols
    n_domains = np.unique(matrix).size
    for domain_i in range(n_domains):
        domain_mask = matrix == domain_i
        domain_size = np.sum(domain_mask)
        matrix[domain_mask] = domain_i + rdn.choice(np.linspace(0.1, 0.9, n_symbols), size=domain_size)
    return matrix

def assign_symbol_to_matrix(matrix: np.ndarray, symbols: dict) -> np.ndarray:
    rdn = np.random.default_rng(seed=RANDOM_SEED)
    
    n_domains = np.unique(matrix).size
    for domain_i in range(n_domains):
        domain_mask = matrix == str(domain_i)
        domain_size = np.sum(domain_mask)
        matrix[domain_mask] = rdn.choice(symbols[domain_i], size=domain_size)
    return matrix

def generate_card_list(n_symbols: int = 5, n_domains: int = 3, n_cards: int = 100, n_symbols_per_card: int = 5) -> List[Card]:
    # Find out the size of a map
    map_side = int(np.sqrt(n_cards))

    # Loop over n_symbols_per_card and generate as many maps
    symbol_map = np.empty((map_side, map_side, n_symbols_per_card), dtype=int)
    for i in range(n_symbols_per_card):
        domain_map = generate_domain_matrix(matrix_side=map_side, n_domains=n_domains)
        # Distribute the values corresponding to n_domains accross the map
        symbol_map[:, :, i] = assign_int_to_matrix(matrix=domain_map, n_symbols=n_symbols)

    # Concatenating the n_symbols_per_card symbols for each card
    return [Card(*map_i) for map_i in symbol_map.reshape(-1, symbol_map.shape[-1])]

def generate_card_list_from_domain_dict(domain_dict: dict, n_cards: int = 100, n_symbols_per_card: int = 5) -> List[Card]:
    # Find out the size of a map
    map_side = int(np.sqrt(n_cards))

    # Loop over n_symbols_per_card and generate as many maps
    symbol_map = np.empty((map_side, map_side, n_symbols_per_card), dtype="U2")
    for i in range(n_symbols_per_card):
        domain_map = generate_domain_matrix(matrix_side=map_side, n_domains=len(domain_dict))
        domain_map = domain_map.astype(str)
        # Distribute the values corresponding to n_domains accross the map
        symbol_map[:, :, i] = assign_symbol_to_matrix(matrix=domain_map, symbols=domain_dict)

    # Concatenating the n_symbols_per_card symbols for each card
    return [Card(*map_i) for map_i in symbol_map.reshape(-1, symbol_map.shape[-1])]