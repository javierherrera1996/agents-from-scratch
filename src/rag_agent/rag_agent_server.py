from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph

from typing import Any, Dict, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import os
import uuid
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

app = FastAPI(title="RAG Agent (FastAPI + A2A)")
# ---------------------------
embeddings = OpenAIEmbeddings()
vector_store = InMemoryVectorStore(embeddings)

llm = ChatOpenAI(model="gpt-4o-mini")



# Only keep post title, headers, and content from the full HTML.
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()



text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # chunk size (characters)
    chunk_overlap=200,  # chunk overlap (characters)
    add_start_index=True,  # track index in original document
)
all_splits = text_splitter.split_documents(docs)
document_ids = vector_store.add_documents(documents=all_splits)



# N.B. for non-US LangSmith endpoints, you may need to specify
# api_url="https://api.smith.langchain.com" in hub.pull.
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "(context goes here)", "question": "(question goes here)"}
).to_messages()



class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
rag_agent = graph_builder.compile()


# ---------------------------
# 2) A2A Minimal Schema
# ---------------------------
class A2AInput(BaseModel):
    # free-form input for your agent; we only require "question"
    question: str

class A2ARequest(BaseModel):
    # Optional unique id supplied by caller
    invocationId: Optional[str] = None
    # Arbitrary metadata (ignored by the agent, echoed back)
    metadata: Optional[Dict[str, Any]] = None
    # The actual input payload expected by the agent
    input: A2AInput

class A2AOutput(BaseModel):
    type: str = "message"
    role: str = "assistant"
    content: str

class A2AResponse(BaseModel):
    invocationId: str
    outputs: List[A2AOutput]
    metadata: Optional[Dict[str, Any]] = None

# ---------------------------
# 3) Health & simple invoke
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/invoke")
def invoke(payload: A2AInput):
    try:
        out = rag_agent.invoke({"question": payload.question})
        return {"answer": out["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------
# 4) A2A endpoint
# ---------------------------
@app.post("/a2a/run", response_model=A2AResponse)
def a2a_run(req: A2ARequest):
    """
    Minimal A2A contract:
    Request:
      {
        "invocationId": "optional-uuid",
        "metadata": {...},          # optional passthrough
        "input": {"question": "..."}
      }

    Response:
      {
        "invocationId": "<echo or generated>",
        "outputs": [{"type":"message","role":"assistant","content":"..."}],
        "metadata": {...}           # optional passthrough echo
      }
    """
    inv_id = req.invocationId or str(uuid.uuid4())
    try:
        result = rag_agent.invoke({"question": req.input.question})
        return A2AResponse(
            invocationId=inv_id,
            outputs=[A2AOutput(content=result["answer"])],
            metadata=req.metadata or {},
        )
    except Exception as e:
        # In A2A you could also return an error output type;
        # keeping it simple with 500 here.
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("rag_agent:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)
