import os
from typing import List, Dict, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
import torch
import json

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi import FastAPI, UploadFile, File, Query, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware

# Create directories
Path("static").mkdir(exist_ok=True)
Path("templates").mkdir(exist_ok=True)
Path("papers").mkdir(exist_ok=True)

# First, let's create the HTML templates
# Create templates/index.html
index_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Research Papers Chat</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        .container {
            display: flex;
            gap: 20px;
            flex: 1;
        }
        .left-panel {
            flex: 1;
            padding: 20px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        .right-panel {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: #fff;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .upload-form {
            margin-bottom: 20px;
        }
        .file-list {
            margin-top: 20px;
        }
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #f9f9f9;
        }
        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 8px;
        }
        .user-message {
            background: #007bff;
            color: white;
            margin-left: 20%;
        }
        .bot-message {
            background: #e9ecef;
            color: black;
            margin-right: 20%;
        }
        .chat-input {
            display: flex;
            padding: 20px;
            background: #f8f9fa;
            border-top: 1px solid #ddd;
        }
        .chat-input input {
            flex: 1;
            padding: 10px;
            margin-right: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        .chat-input button {
            padding: 10px 20px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .chat-input button:hover {
            background: #0056b3;
        }
        .upload-button {
            background: #28a745;
            color: white;
            padding: 10px 20px;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        .upload-button:hover {
            background: #218838;
        }
        .file-item {
            padding: 10px;
            background: #fff;
            margin: 5px 0;
            border-radius: 4px;
            border: 1px solid #ddd;
        }
        .sources {
            font-size: 0.8em;
            color: #666;
            margin-top: 5px;
            padding-top: 5px;
            border-top: 1px solid #ddd;
        }
    </style>
</head>
<body>
    <h1>Research Papers Chat Interface</h1>
    <div class="container">
        <div class="left-panel">
            <h2>Upload Papers</h2>
            <form class="upload-form" id="uploadForm">
                <input type="file" id="pdfFile" accept=".pdf" multiple>
                <button type="submit" class="upload-button">Upload</button>
            </form>
            <div class="file-list" id="fileList">
                <h3>Uploaded Papers</h3>
                <!-- Files will be listed here -->
            </div>
        </div>
        <div class="right-panel">
            <div class="chat-messages" id="chatMessages">
                <!-- Messages will appear here -->
            </div>
            <div class="chat-input">
                <input type="text" id="messageInput" placeholder="Ask a question about the papers...">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
    </div>

    <script>
        let ws = null;

        function connectWebSocket() {
            ws = new WebSocket('ws://localhost:8000/ws');
            
            ws.onmessage = function(event) {
                const response = JSON.parse(event.data);
                displayMessage(response.text, 'bot', response.sources);
            };

            ws.onclose = function() {
                setTimeout(connectWebSocket, 1000);  // Reconnect after 1 second
            };
        }

        connectWebSocket();

        document.getElementById('uploadForm').onsubmit = async function(e) {
            e.preventDefault();
            const files = document.getElementById('pdfFile').files;
            const formData = new FormData();
            
            for (let file of files) {
                formData.append('file', file);
                
                try {
                    const response = await fetch('/papers/upload', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (response.ok) {
                        const result = await response.json();
                        if (result.success) {
                            displayMessage(`Successfully uploaded ${file.name}`, 'bot');
                            updateFileList();
                        }
                    }
                } catch (error) {
                    console.error('Error uploading file:', error);
                    displayMessage(`Error uploading ${file.name}`, 'bot');
                }
            }
        };

        async function updateFileList() {
            try {
                const response = await fetch('/papers/list');
                const files = await response.json();
                const fileList = document.getElementById('fileList');
                fileList.innerHTML = '<h3>Uploaded Papers</h3>';
                files.forEach(file => {
                    const fileItem = document.createElement('div');
                    fileItem.className = 'file-item';
                    fileItem.textContent = file;
                    fileList.appendChild(fileItem);
                });
            } catch (error) {
                console.error('Error updating file list:', error);
            }
        }

        function displayMessage(text, sender, sources = null) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}-message`;
            messageDiv.textContent = text;
            
            if (sources && sources.length > 0) {
                const sourcesDiv = document.createElement('div');
                sourcesDiv.className = 'sources';
                sourcesDiv.innerHTML = '<strong>Sources:</strong><br>' + 
                    sources.map(source => 
                        `${source.title} (Page ${source.page})`
                    ).join('<br>');
                messageDiv.appendChild(sourcesDiv);
            }
            
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (message && ws && ws.readyState === WebSocket.OPEN) {
                displayMessage(message, 'user');
                ws.send(message);
                input.value = '';
            }
        }

        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });

        // Initial file list update
        updateFileList();
    </script>
</body>
</html>
"""

# Save the template
with open("templates/index.html", "w") as f:
    f.write(index_html)

@dataclass
class PaperMetadata:
    title: str
    authors: List[str]
    publication_date: datetime
    publication_venue: str
    file_path: str
    num_pages: int

class ResearchPaperRAG:
    def __init__(self, papers_dir: str = "papers", db_dir: str = "paper_db"):
        self.papers_dir = Path(papers_dir)
        self.papers_dir.mkdir(exist_ok=True)
        
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
            )
        except Exception as e:
            print(f"Warning: Error initializing GPU embeddings, falling back to CPU: {e}")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-mpnet-base-v2",
                model_kwargs={'device': 'cpu'}
            )
        
        self.vector_store = Chroma(
            persist_directory=db_dir,
            embedding_function=self.embeddings
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        
        self.paper_metadata: Dict[str, PaperMetadata] = {}
        
    async def add_paper(self, file: UploadFile) -> bool:
        try:
            file_path = self.papers_dir / file.filename
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            loader = PyPDFLoader(str(file_path))
            pages = loader.load()
            
            metadata = PaperMetadata(
                title=file.filename.replace(".pdf", ""),
                authors=[],
                publication_date=datetime.now(),
                publication_venue="",
                file_path=str(file_path),
                num_pages=len(pages)
            )
            self.paper_metadata[file.filename] = metadata
            
            chunks = self.text_splitter.split_documents(pages)
            
            texts = [chunk.page_content for chunk in chunks]
            metadatas = [{
                "source": file.filename,
                "page": chunk.metadata.get("page", 0),
                "chunk_index": i
            } for i, chunk in enumerate(chunks)]
            
            self.vector_store.add_texts(texts=texts, metadatas=metadatas)
            
            return True
        except Exception as e:
            print(f"Error adding paper: {e}")
            return False
    
    def get_response(self, question: str) -> Dict:
        """Get a response to a question using the paper collection."""
        results = self.vector_store.similarity_search_with_score(
            question,
            k=3
        )
        
        sources = []
        relevant_text = []
        for doc, score in results:
            metadata = self.paper_metadata.get(doc.metadata["source"])
            if metadata:
                sources.append({
                    "title": metadata.title,
                    "page": doc.metadata["page"]
                })
                relevant_text.append(doc.page_content)
        
        # For now, we'll just return the most relevant text
        # In a production system, you'd want to use an LLM to generate a proper answer
        response = f"Based on the papers, here's what I found:\n\n{relevant_text[:5]}"
        
        return {
            "text": response,
            "sources": sources
        }

# FastAPI application
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize RAG system
rag_system = ResearchPaperRAG()

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/papers/upload")
async def upload_paper(file: UploadFile = File(...)):
    success = await rag_system.add_paper(file)
    return {"success": success}

@app.get("/papers/list")
async def list_papers():
    papers = [f.name for f in Path("papers").glob("*.pdf")]
    return papers

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        while True:
            question = await websocket.receive_text()
            response = rag_system.get_response(question)
            await websocket.send_json(response)
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    print("Server starting up...")
    print("Access the web interface at: http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)