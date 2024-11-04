# Python RAG App

ISABE project - Python RAG App 

The main purpose of this project is to scan reseach papers, and the user can ask questions related to those papers. The response will be generated along with linking the page of the specific PDF of the research paper to make the reseach process easier.

## How to Use
- Install [Ollama](https://ollama.com/) and the [Mistral model](https://ollama.com/library/mistral)
- Run Ollama using `ollama serve`
- Create an environment for the app using venv (`python -m venv env`) and activate it (`env/bin/activate`)
- Save the `main.py` file in the same environment directory
- Install python dependencies - `pip install torch chromadb langchain fastapi uvicorn sentence-transformers pypdf jinja2 "langchain[chromadb]" "fastapi[all]"`
- Run `main.py`
- Open `http://localhost:8000/` on your browser, the application should be running

## Working
This application utlizes Ollama + Mistral model to run the AI on your local machine. It uses it to scan the PDFs using Langchain (by splitting it up and using a vector DB). This is called a Retrieval-Augmented Generation (RAG), which is then combined with a large language model (LLM) to provide responses.

- **Stage 1** - The users upload PDFs and then each PDF is parsed using PyPDFLoader to extract text by pages. This allows us to reference specific pages.
- **Stage 2** - After parsing, text from each PDF is embedded into vector representations, which enables semantic search. This is done using HuggingFaceEmbeddings. This is also stored in a scalable vector database like Chroma.
- **Stage 3** - The RAG searches, filters and sorts the texts according to the query. We utilize FastAPI for this.
- **Stage 4** - The text is given to the LLM (Ollama with Mistral model in this case, but any model like Gemma, GPT, Llama etc can be used)

## Demo Video
Youtube Link - https://youtu.be/v2t2ERfuQZI

## To-Do
- [ ] Make the upload time shorter
- [ ] Improve response style
- [ ] Integrate better with Ollama
