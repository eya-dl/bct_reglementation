# test_decomposer.py
from agenticIA.decomposer import is_composite, split_into_subquestions
test_cases = [
    # (query, doit_etre_composite)
    ("Quelles sont les conditions et modalités de rapatriement des devises ?", False),
    ("Quelles sont les conditions de rapatriement et quelles sont les sanctions en cas de non-respect ?", True),
    ("Comment déclarer un compte en devises et quand faut-il le faire ?", True),
    ("Quel est le plafond autorisé pour les opérations de change manuel ?", False),
    ("Quelles sont les obligations des banques ? Quelles sont celles des résidents ?", True),
    ("Est-ce que les non-résidents peuvent ouvrir un compte en devises ?", False),
]

for query, expected in test_cases:
    result = is_composite(query)
    status = "OK" if result == expected else "ÉCHEC"
    print(f"[{status}] composite={result} (attendu={expected}) — {query}")