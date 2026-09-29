from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import sqlite3
import os
import hashlib
import struct
import numpy as np
import ollama
import faiss
from pydantic import BaseModel
from typing import List, Optional
import json
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
DB = "rag.db"
CONFIG = "config"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL = "gemma3:latest"

# Models
class QueryRequest(BaseModel):
    question: str

class ConfigUpdate(BaseModel):
    files: List[str]

class StartupResponse(BaseModel):
    needs_update: bool
    added: List[str]
    removed: List[str]
    modified: List[str]

class ContextItem(BaseModel):
    file: str
    start: int
    end: int
    preview: str

class QueryResponse(BaseModel):
    answer: str
    context_used: List[ContextItem]

# Database setup
def init_db():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS docs (
            id INTEGER PRIMARY KEY,
            path TEXT UNIQUE,
            hash TEXT,
            last_modified REAL
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            doc_id INTEGER,
            text TEXT,
            start_line INTEGER,
            end_line INTEGER,
            FOREIGN KEY(doc_id) REFERENCES docs(id)
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS embeddings_map (
            chunk_id INTEGER,
            faiss_id INTEGER,
            FOREIGN KEY(chunk_id) REFERENCES chunks(id)
        )
    """)
    con.commit()
    con.close()

# File operations
def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def load_config():
    if not os.path.exists(CONFIG):
        return []
    with open(CONFIG, "r") as f:
        return [os.path.expanduser(line.strip()) for line in f if line.strip()]

def save_config(files):
    with open(CONFIG, "w") as f:
        for file in files:
            f.write(file + "\n")

# Chunking
def chunk_file(text, file_path):
    chunks = []
    lines = text.splitlines()
    
    if not lines:
        return chunks
    
    # Simple chunking: by fixed number of lines with overlap
    chunk_size = 20
    overlap = 5
    
    i = 0
    while i < len(lines):
        end = min(i + chunk_size, len(lines))
        chunk_text = '\n'.join(lines[i:end])
        
        chunks.append({
            'text': chunk_text,
            'start': i,
            'end': end - 1
        })
        
        i += (chunk_size - overlap)
    
    return chunks

# FAISS operations
def get_faiss_index():
    if os.path.exists("faiss.index"):
        return faiss.read_index("faiss.index")
    else:
        # nomic-embed-text has 768 dimensions
        return faiss.IndexFlatIP(768)

def save_faiss_index(index):
    faiss.write_index(index, "faiss.index")

# Embedding
def get_embedding(text):
    try:
        result = ollama.embeddings(model=EMBED_MODEL, prompt=text)
        return np.array(result["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"Embedding error: {e}")
        # Return a zero vector as fallback
        return np.zeros(768, dtype=np.float32)

# API Endpoints
@app.get("/")
async def serve_frontend():
    return FileResponse("frontend/index.html")

@app.get("/startup")
async def startup_check():
    """Check if reindexing is needed"""
    try:
        config_files = set(load_config())
        
        con = sqlite3.connect(DB)
        cur = con.cursor()
        cur.execute("SELECT path, hash FROM docs")
        db_files = {row[0]: row[1] for row in cur.fetchall()}
        con.close()
        
        added = config_files - set(db_files.keys())
        removed = set(db_files.keys()) - config_files
        
        modified = []
        for file in config_files & set(db_files.keys()):
            if os.path.exists(file) and file_hash(file) != db_files[file]:
                modified.append(file)
        
        return StartupResponse(
            needs_update=bool(added or removed or modified),
            added=list(added),
            removed=list(removed),
            modified=modified
        )
    except Exception as e:
        print(f"Startup check error: {e}")
        return StartupResponse(needs_update=False, added=[], removed=[], modified=[])

@app.post("/reindex")
async def reindex():
    """Reindex all files"""
    try:
        init_db()
        config_files = load_config()
        
        con = sqlite3.connect(DB)
        cur = con.cursor()
        
        # Clear existing data
        cur.execute("DELETE FROM chunks")
        cur.execute("DELETE FROM docs")
        cur.execute("DELETE FROM embeddings_map")
        con.commit()
        
        # Create new FAISS index
        index = faiss.IndexFlatIP(768)
        all_embeddings = []
        chunk_mappings = []
        
        total_chunks = 0
        
        for file_path in config_files:
            if not os.path.exists(file_path):
                print(f"File not found: {file_path}")
                continue
                
            try:
                print(f"Processing: {file_path}")
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                
                file_hash_val = file_hash(file_path)
                
                # Insert document
                cur.execute(
                    "INSERT OR REPLACE INTO docs (path, hash, last_modified) VALUES (?, ?, ?)",
                    (file_path, file_hash_val, os.path.getmtime(file_path))
                )
                doc_id = cur.lastrowid
                
                # Chunk file
                chunks = chunk_file(text, file_path)
                print(f"  Created {len(chunks)} chunks")
                
                for chunk in chunks:
                    cur.execute(
                        "INSERT INTO chunks (doc_id, text, start_line, end_line) VALUES (?, ?, ?, ?)",
                        (doc_id, chunk['text'], chunk['start'], chunk['end'])
                    )
                    chunk_id = cur.lastrowid
                    
                    # Generate embedding
                    embedding = get_embedding(chunk['text'])
                    all_embeddings.append(embedding)
                    chunk_mappings.append(chunk_id)
                    
                    total_chunks += 1
        
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        # Add all embeddings to FAISS
        if all_embeddings:
            embeddings_array = np.array(all_embeddings)
            print(f"Adding {len(embeddings_array)} embeddings to FAISS index")
            index.add(embeddings_array)
            save_faiss_index(index)
            
            # Store chunk to FAISS ID mapping
            for faiss_id, chunk_id in enumerate(chunk_mappings):
                cur.execute(
                    "INSERT INTO embeddings_map (chunk_id, faiss_id) VALUES (?, ?)",
                    (chunk_id, faiss_id)
                )
        
        con.commit()
        con.close()
        
        return {"status": "success", "message": f"Indexed {len(config_files)} files, created {total_chunks} chunks"}
    
    except Exception as e:
        print(f"Reindex error: {e}")
        return {"status": "error", "message": str(e)}

@app.post("/query")
async def query(request: QueryRequest):
    """Handle RAG query"""
    try:
        if not os.path.exists("faiss.index"):
            raise HTTPException(status_code=400, detail="Index not built. Please reindex first.")
        
        print(f"Query: {request.question}")
        
        # Embed query
        query_embedding = get_embedding(request.question)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Search FAISS
        index = get_faiss_index()
        k = min(3, index.ntotal)  # Don't ask for more than available
        if k == 0:
            raise HTTPException(status_code=400, detail="No documents in index")
            
        print(f"Searching {k} nearest neighbors from {index.ntotal} total vectors")
        D, I = index.search(query_embedding, k=k)
        
        # Get chunk details
        con = sqlite3.connect(DB)
        cur = con.cursor()
        
        context_chunks = []
        context_used = []
        
        for i, faiss_id in enumerate(I[0]):
            if faiss_id < 0:  # Invalid ID
                continue
                
            # Get chunk_id from mapping
            cur.execute("SELECT chunk_id FROM embeddings_map WHERE faiss_id = ?", (int(faiss_id),))
            mapping_result = cur.fetchone()
            
            if not mapping_result:
                continue
                
            chunk_id = mapping_result[0]
            
            cur.execute("""
                SELECT c.text, c.start_line, c.end_line, d.path 
                FROM chunks c 
                JOIN docs d ON c.doc_id = d.id 
                WHERE c.id = ?
            """, (chunk_id,))
            
            result = cur.fetchone()
            if result:
                text, start, end, path = result
                # Format context with file info
                formatted_context = f"[file: {path} | lines {start}-{end}]\n{text}"
                context_chunks.append(formatted_context)
                context_used.append(ContextItem(
                    file=path,
                    start=start,
                    end=end,
                    preview=text[:200] + "..." if len(text) > 200 else text
                ))
        
        if not context_chunks:
            raise HTTPException(status_code=400, detail="No context found for query")
        
        # Build prompt
        context_text = "\n\n".join(context_chunks)
        prompt = f"""Context information:
{context_text}

Based on the above context, please answer this question:

Question: {request.question}

Answer:"""
        
        print(f"Context chunks: {len(context_chunks)}")
        print(f"Prompt length: {len(prompt)}")
        
        # Get LLM response
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=False
        )
        
        answer = response["message"]["content"]
        
        # Save to history
        cur.execute(
            "INSERT INTO history (question, answer) VALUES (?, ?)",
            (request.question, answer)
        )
        con.commit()
        con.close()
        
        return QueryResponse(answer=answer, context_used=context_used)
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

@app.get("/config")
async def get_config():
    return {"files": load_config()}

@app.post("/config")
async def update_config(config: ConfigUpdate):
    save_config(config.files)
    return {"status": "success"}

@app.delete("/history")
async def clear_history():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("DELETE FROM history")
    con.commit()
    con.close()
    return {"status": "success"}

@app.get("/history")
async def get_history():
    con = sqlite3.connect(DB)
    cur = con.cursor()
    cur.execute("SELECT question, answer, timestamp FROM history ORDER BY timestamp DESC LIMIT 50")
    history = [{"question": row[0], "answer": row[1], "timestamp": row[2]} for row in cur.fetchall()]
    con.close()
    return history

# Serve static files
app.mount("/", StaticFiles(directory="frontend"), name="frontend")

if __name__ == "__main__":
    import uvicorn
    # Initialize database
    init_db()
    print("Starting RAG server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")