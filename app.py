from flask import Flask, render_template, request
from ddgs import DDGS
from sentence_transformers import SentenceTransformer, util
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from functools import lru_cache
import torch

app = Flask(__name__)

# Detectar dispositivo: GPU si está disponible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")

# Modelo ligero
model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

# Cache limitada
MAX_CACHE = 50

# Función para extraer texto de HTML
def extract_text(html):
    soup = BeautifulSoup(html, "html.parser")
    return " ".join(p.get_text() for p in soup.find_all("p"))

# Descarga asíncrona de páginas
async def fetch(session, url):
    try:
        async with session.get(url, timeout=5) as resp:
            html = await resp.text()
            text = extract_text(html)
            return text if text else ""
    except:
        return ""

async def fetch_all(urls):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch(session, url) for url in urls]
        return await asyncio.gather(*tasks)

# Búsqueda multi-motor gratuita
def buscar_multi_motor(query, max_results=50):
    resultados = []

    # DuckDuckGo
    with DDGS() as ddgs:
        resultados += list(ddgs.text(query, max_results=max_results))

    # Motor adicional simulado (ej: Qwant / Startpage)
    # Aquí puedes implementar scraping real
    # Ejemplo conceptual: resultados ficticios
    # resultados += [{"href": "https://example.com", "body": "Texto ejemplo", "title": "Ejemplo"}]

    # Eliminar duplicados por URL
    seen = set()
    resultados_unicos = []
    for r in resultados:
        href = r.get("href")
        if href and href not in seen:
            resultados_unicos.append(r)
            seen.add(href)

    return resultados_unicos

# Función principal de búsqueda con LRU cache
@lru_cache(maxsize=MAX_CACHE)
def buscar_contenido(query, max_results=50):
    resultados = buscar_multi_motor(query, max_results=max_results)

    urls = [r.get("href") for r in resultados if r.get("href")]

    # Descargar páginas asíncronamente
    textos = asyncio.run(fetch_all(urls))

    # Si algún texto está vacío, usar body de DDGS
    for i, r in enumerate(resultados):
        if not textos[i]:
            textos[i] = r.get("body", "")

    # Embeddings batch
    query_emb = model.encode(query, convert_to_tensor=True)
    doc_embs = model.encode(textos, convert_to_tensor=True, batch_size=32)
    cos_scores = util.cos_sim(query_emb, doc_embs)[0]

    # Ordenar resultados por relevancia
    resultados_ordenados = sorted(
        zip(urls, textos, cos_scores),
        key=lambda x: x[2],
        reverse=True
    )

    return resultados_ordenados

# Ruta principal con paginado
@app.route("/", methods=["GET", "POST"])
def index():
    query = ""
    resultados = []

    # Obtener query de POST o GET
    if request.method == "POST":
        query = request.form["query"]
    else:
        query = request.args.get("query", "")

    # Obtener página, con paréntesis corregido
    page = int(request.args.get("page", 1))

    if query:
        resultados = buscar_contenido(query)

    # Paginado
    per_page = 10
    start = (page - 1) * per_page
    end = start + per_page
    total_pages = (len(resultados) + per_page - 1) // per_page
    resultados_pagina = resultados[start:end]

    return render_template(
        "index.html",
        query=query,
        resultados=resultados_pagina,
        page=page,
        total_pages=total_pages
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
