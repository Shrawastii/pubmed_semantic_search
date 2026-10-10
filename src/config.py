import os, random
import numpy as np
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("NCBI_API_KEY")
EMAIL = os.getenv("NCBI_EMAIL")
SEED = 42

# Use the exact query that gave you ~3,500 results in the count check
QUERY = (
    '"Encephalitis"[MeSH Terms] AND hasabstract AND English[lang] '
    'AND humans[MeSH Terms] AND 2024/10/09:2026/10/09[dp]'
)

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)