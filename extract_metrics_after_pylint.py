import json
import os
import subprocess
import sys
from collections import defaultdict, Counter

# ConfiguraÃ§Ã£o
PROJETO = "./src" # diretÃ³rio do cÃ³digo fonte
PASTA   = "metrics-after-pylint" # NÃ£o altere o nome dessa pasta, os relatÃ³rios vÃ£o ser salvos nela.

os.makedirs(PASTA, exist_ok=True)

# Roda o Pylint
print(f"Rodando pylint em {PROJETO}...")

resultado = subprocess.run(
    ["pylint", PROJETO, "--output-format=json", "--score=y"],
    capture_output=True,
    text=True,
    encoding="utf-8",
)

# Salva o JSON bruto
caminho_json = os.path.join(PASTA, "pylint_depois.json")
with open(caminho_json, "w", encoding="utf-8") as f:
    f.write(resultado.stdout)
print(f"JSON completo salvo em: {caminho_json}")

# Processa mensagens
try:
    mensagens = json.loads(resultado.stdout)
except json.JSONDecodeError:
    print("Erro ao processar JSON do Pylint.")
    sys.exit(1)

if not mensagens:
    print("Nenhuma mensagem encontrada.")
    sys.exit(0)

# Salva JSONs por categoria
por_tipo = defaultdict(list)
for msg in mensagens:
    tipo = msg.get("type", "unknown")
    por_tipo[tipo].append(msg)

tipos_nomes = {
    "convention": "pylint_convention_depois.json",
    "refactor":   "pylint_refactor_depois.json",
    "warning":    "pylint_warning_depois.json",
    "error":      "pylint_error_depois.json",
    "fatal":      "pylint_fatal_depois.json",
}

for tipo, nome_arquivo in tipos_nomes.items():
    caminho = os.path.join(PASTA, nome_arquivo)
    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(por_tipo.get(tipo, []), f, indent=2, ensure_ascii=False)
    print(f"{len(por_tipo.get(tipo, [])):>5} mensagens â†’ {caminho}")

print(f"\nTotal: {len(mensagens)} mensagens encontradas.")

# Ranking da categoria refactor
mensagens_refactor = [msg for msg in mensagens if msg.get("type") == "refactor"]
contagem_simbolos = Counter(msg["symbol"] for msg in mensagens_refactor)
caminho_ranking = os.path.join(PASTA, "pylint_ranking_smells_depois.json")
with open(caminho_ranking, "w", encoding="utf-8") as f:
    json.dump(
        [{"simbolo": s, "ocorrencias": t} for s, t in contagem_simbolos.most_common()],
        f,
        indent=2,
        ensure_ascii=False,
    )
print(f"Ranking de sÃ­mbolos salvo em: {caminho_ranking}")

# Arquivos com mais problemas
por_arquivo = defaultdict(lambda: defaultdict(int))
for msg in mensagens:
    path = msg.get("path", "desconhecido")
    tipo = msg.get("type", "unknown")
    por_arquivo[path][tipo] += 1
    por_arquivo[path]["total"] += 1

arquivos_ordenados = sorted(
    [
        {"arquivo": path, **contagens}
        for path, contagens in por_arquivo.items()
    ],
    key=lambda x: x["total"],
    reverse=True,
)
caminho_arquivos = os.path.join(PASTA, "pylint_arquivos_criticos_depois.json")
with open(caminho_arquivos, "w", encoding="utf-8") as f:
    json.dump(arquivos_ordenados, f, indent=2, ensure_ascii=False)
print(f"Arquivos crÃ­ticos salvo em: {caminho_arquivos}")

# DistribuiÃ§Ã£o por categoria
total = len(mensagens)
distribuicao = [
    {
        "categoria": tipo,
        "ocorrencias": len(msgs),
        "percentual": round(len(msgs) / total * 100, 2),
    }
    for tipo, msgs in por_tipo.items()
]
distribuicao.sort(key=lambda x: x["ocorrencias"], reverse=True)
caminho_dist = os.path.join(PASTA, "pylint_distribuicao_categorias_depois.json")
with open(caminho_dist, "w", encoding="utf-8") as f:
    json.dump(distribuicao, f, indent=2, ensure_ascii=False)
print(f"DistribuiÃ§Ã£o por categoria salva em: {caminho_dist}")

print("\nConcluÃ­do.")