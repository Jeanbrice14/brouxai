FROM python:3.11-slim

WORKDIR /app

# nodejs/npm : requis par DataAgent/MetadataAgent pour lancer le Power BI Modeling MCP
# Server via `npx` (app/services/powerbi_local_mcp.py) — absent de l'image de base. Notez
# que ce serveur MCP refuse toute connexion Analysis Services locale depuis un runtime .NET
# Linux ("This feature is supported for a .NET Core client only on Windows systems.") —
# nodejs reste installé pour permettre au serveur de démarrer, mais connect_to_desktop_file
# échouera toujours depuis ce conteneur (voir README.md, section RAG schéma sémantique).
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential nodejs npm \
    && rm -rf /var/lib/apt/lists/*

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
