import csv
import json
import os
import subprocess

def normalizar_caminho(path):
    return path.replace("\\", "/")

# Gera JSONs via Radon
def rodar_radon(comando):
    resultado = subprocess.run(comando, capture_output=True, text=True, encoding="utf-8")
    return json.loads(resultado.stdout)

print("Rodando radon cc...")
cc_data  = rodar_radon(["radon", "cc",  "./src", "-j"])
print("Rodando radon mi...")
mi_data  = rodar_radon(["radon", "mi",  "./src", "-j"])
print("Rodando radon hal...")
hal_data = rodar_radon(["radon", "hal", "./src", "-j"])
print("Rodando radon raw...")
raw_data = rodar_radon(["radon", "raw", "./src", "-j"])

# NÃ£o altere o nome dessa pasta, os relatÃ³rios vÃ£o ser salvos nela.
pasta = "metrics-after-radon"  
os.makedirs(pasta, exist_ok=True)

with open(os.path.join(pasta, "cc_depois.json"),  "w", encoding="utf-8") as f:
    json.dump(cc_data,  f, indent=2)
with open(os.path.join(pasta, "mi_depois.json"),  "w", encoding="utf-8") as f:
    json.dump(mi_data,  f, indent=2)
with open(os.path.join(pasta, "hal_depois.json"), "w", encoding="utf-8") as f:
    json.dump(hal_data, f, indent=2)
with open(os.path.join(pasta, "raw_depois.json"), "w", encoding="utf-8") as f:
    json.dump(raw_data, f, indent=2)

print("JSONs salvos.")

HAL_FIELDS = ["h1", "h2", "N1", "N2", "vocabulary", "length",
              "calculated_length", "volume", "difficulty",
              "effort", "time", "bugs"]

RAW_FIELDS = ["loc", "lloc", "sloc", "comments", "multi", "blank",
              "single_comments"]

rows_cc          = []
rows_cc_arquivo  = []
rows_mi          = []
rows_hal_arquivo = []
rows_hal_funcao  = []
rows_raw         = []

# MI por arquivo 
for arquivo, dados in mi_data.items():
    rows_mi.append({
        "arquivo": normalizar_caminho(arquivo),
        "mi":      round(dados["mi"], 4),
        "rank_mi": dados["rank"],
    })

# CC por funÃ§Ã£o e por arquivo
def extrair_funcoes(blocos, arquivo):
    resultado = []
    for bloco in blocos:
        if bloco["type"] == "class":
            continue
        resultado.append({
            "arquivo":    normalizar_caminho(arquivo),
            "tipo":       bloco["type"],
            "classe":     bloco.get("classname") or "",
            "nome":       bloco["name"],
            "rank_cc":    bloco["rank"],
            "complexity": bloco["complexity"],
            "linha_ini":  bloco["lineno"],
            "linha_fim":  bloco["endline"],
        })
        if bloco.get("closures"):
            resultado += extrair_funcoes(bloco["closures"], arquivo)
    return resultado

vistas = set()

for arquivo, blocos in cc_data.items():
    funcoes_arquivo = []
    for bloco in extrair_funcoes(blocos, arquivo):
        chave = (bloco["arquivo"], bloco["nome"], bloco["linha_ini"])
        if chave in vistas:
            continue
        vistas.add(chave)
        rows_cc.append(bloco)
        funcoes_arquivo.append(bloco)

    if funcoes_arquivo:
        complexidades = [f["complexity"] for f in funcoes_arquivo]
        pior = max(funcoes_arquivo, key=lambda x: x["complexity"])
        rows_cc_arquivo.append({
            "arquivo":        normalizar_caminho(arquivo),
            "funcoes":        len(funcoes_arquivo),
            "cc_media":       round(sum(complexidades) / len(complexidades), 2),
            "cc_max":         max(complexidades),
            "cc_soma":        sum(complexidades),
            "pior_rank":      pior["rank_cc"],
            "pior_classe":    pior["classe"],
            "pior_funcao":    pior["nome"],
            "pior_linha_ini": pior["linha_ini"],
        })
    else:
        rows_cc_arquivo.append({
            "arquivo":        normalizar_caminho(arquivo),
            "funcoes":        0,
            "cc_media":       "",
            "cc_max":         "",
            "cc_soma":        "",
            "pior_rank":      "",
            "pior_classe":    "",
            "pior_funcao":    "",
            "pior_linha_ini": "",
        })

# Halstead por arquivo e por funÃ§Ã£o
vistos_hal = set()

for arquivo, dados in hal_data.items():
    arq_norm = normalizar_caminho(arquivo)

    t = dados["total"]
    row = {"arquivo": arq_norm, "escopo": "arquivo", "nome": ""}
    for field in HAL_FIELDS:
        val = t.get(field, 0) or 0
        row[field] = round(val, 4)
    rows_hal_arquivo.append(row)

    for nome_func, func_hal in dados.get("functions", {}).items():
        chave = (arq_norm, nome_func)
        nome_final = nome_func
        if chave in vistos_hal:
            count = sum(1 for k in vistos_hal if k[0] == arq_norm and k[1].startswith(nome_func))
            nome_final = f"{nome_func}_{count}"
        vistos_hal.add((arq_norm, nome_final))

        row = {"arquivo": arq_norm, "escopo": "funcao", "nome": nome_final}
        for field in HAL_FIELDS:
            val = func_hal.get(field, 0) or 0
            row[field] = round(val, 4)
        rows_hal_funcao.append(row)

# Raw por arquivo e total
total_raw = {field: 0 for field in RAW_FIELDS}

for arquivo, dados in raw_data.items():
    row = {"arquivo": normalizar_caminho(arquivo)}
    for field in RAW_FIELDS:
        val = dados.get(field, 0) or 0
        row[field] = val
        total_raw[field] += val
    rows_raw.append(row)

rows_raw.append({"arquivo": "TOTAL", **total_raw})

# Exporta CSVs
def salvar_csv(nome, linhas, ordenar_por=None):
    if not linhas:
        print(f"Sem dados para {nome}")
        return
    if ordenar_por:
        linhas = sorted(linhas, key=lambda x: (x[ordenar_por] == "", x[ordenar_por]))
    caminho = os.path.join(pasta, nome)
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=linhas[0].keys())
        writer.writeheader()
        writer.writerows(linhas)
    print(f"{len(linhas):>5} linhas â†’ {caminho}")

salvar_csv("mi_por_arquivo_depois.csv",   rows_mi,          ordenar_por="mi")
salvar_csv("cc_por_funcao_depois.csv",    rows_cc,          ordenar_por="complexity")
salvar_csv("cc_por_arquivo_depois.csv",   rows_cc_arquivo,  ordenar_por="cc_media")
salvar_csv("hal_por_arquivo_depois.csv",  rows_hal_arquivo, ordenar_por="effort")
salvar_csv("hal_por_funcao_depois.csv",   rows_hal_funcao)
salvar_csv("raw_por_arquivo_e_total_depois.csv",  rows_raw,         ordenar_por="sloc")

print("\nConcluÃ­do.")