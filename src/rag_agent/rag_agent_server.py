from __future__ import annotations

import os
import uuid
import logging
from typing import Any, Dict, Optional, List

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import bs4
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain import hub
from langgraph.graph import START, StateGraph

# ----------------------------------
# Logging
# ----------------------------------
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("rag-server")

# ----------------------------------
# FastAPI
# ----------------------------------
app = FastAPI(title="RAG Agent (FastAPI + A2A)")

# Permite probar desde localhost o frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # restringe en prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------------
# LLMs & Vector store
# ----------------------------------
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

# Chat model (ajusta tiempo y temperatura si quieres)
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Loader (solo mantiene título, header, contenido del post)
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()

# Split en chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True,
)
all_splits = text_splitter.split_documents(docs)

# Indexa
_ = vector_store.add_documents(documents=all_splits)

# Prompt desde LangChain Hub (con fallback)
try:
    prompt = hub.pull("rlm/rag-prompt")
    _ = prompt.invoke({"context": "(context)", "question": "(question)"}).to_messages()
    log.info("Prompt cargado desde LangChain Hub.")
except Exception as e:
    log.warning(f"No se pudo cargar prompt del Hub, usando fallback. Detalle: {e}")
    from langchain_core.prompts import ChatPromptTemplate
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "Eres un asistente que responde usando solo el CONTEXTO dado. "
             "Si no está en el contexto, di que no lo sabes. "
             "Responde de forma concisa en español."),
            ("human", "PREGUNTA: {question}\n\nCONTEXTO:\n{context}")
        ]
    )

# ----------------------------------
# LangGraph RAG
# ----------------------------------
class State(dict):
    question: str
    context: List[Document]
    answer: str
    citations: List[str]

def retrieve(state: State) -> Dict[str, Any]:
    # Top-k configurable si quieres (aquí 4)
    retrieved_docs = vector_store.similarity_search(state["question"], k=4)
    return {"context": retrieved_docs}

def generate(state: State) -> Dict[str, Any]:
    docs = state["context"]
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)

    # Extrae citas/urls de metadatos si existen
    citations: List[str] = []
    for d in docs:
        src = d.metadata.get("source") or d.metadata.get("loc") or "desconocido"
        if src not in citations:
            citations.append(src)

    return {"answer": response.content, "citations": citations}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
rag_agent = graph_builder.compile()

# ----------------------------------
# A2A schema
# ----------------------------------
class A2AInput(BaseModel):
    question: str

class A2ARequest(BaseModel):
    invocationId: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    input: A2AInput

class A2AOutput(BaseModel):
    type: str = "message"
    role: str = "assistant"
    content: str

class A2AMetadata(BaseModel):
    citations: Optional[List[str]] = None
    echo: Optional[Dict[str, Any]] = None

class A2AResponse(BaseModel):
    invocationId: str
    outputs: List[A2AOutput]
    metadata: Optional[A2AMetadata] = None

# ----------------------------------
# Endpoints
# ----------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/invoke")
def invoke(payload: A2AInput):
    try:
        out = rag_agent.invoke({"question": payload.question})
        return {"answer": out["answer"], "citations": out.get("citations", [])}
    except Exception as e:
        log.exception("Error en /invoke")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/a2a/run", response_model=A2AResponse)
def a2a_run(req: A2ARequest):
    """
    Minimal A2A contract:
      Request:
        {
          "invocationId": "optional-uuid",
          "metadata": {...},          # passthrough opcional
          "input": {"question": "..."}
        }
    """
    inv_id = req.invocationId or str(uuid.uuid4())
    try:
        result = rag_agent.invoke({"question": req.input.question})
        return A2AResponse(
            invocationId=inv_id,
            outputs=[A2AOutput(content=result["answer"])],
            metadata=A2AMetadata(
                citations=result.get("citations", []),
                echo=req.metadata or {},
            ),
        )
    except Exception as e:
        log.exception("Error en /a2a/run")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # OJO: usa el objeto `app` directamente para evitar depender del nombre del archivo
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
