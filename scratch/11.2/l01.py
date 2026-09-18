print("=== A: first call ===")
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()                      # LLM_BASE_URL, LLM_API_KEY, LLM_MODEL from .env
model = ChatOpenAI(base_url=os.environ["LLM_BASE_URL"], api_key=os.environ["LLM_API_KEY"],
                   model=os.environ["LLM_MODEL"], temperature=0, max_tokens=200)

resp = model.invoke("A Quillstack customer writes: 'my 2FA code is rejected every time, "
                    "my phone clock is synced'. Is this a how-to question or a bug report? One sentence.")
print(type(resp).__name__)
print(resp.content)
print(resp.usage_metadata)

print("=== B: chain ===")
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are Resolve, Quillstack's support assistant. Answer only from the context. "
               "If the context does not answer, say so."),
    ("human", "Context:\n{context}\n\nQuestion: {question}"),
])
chain = prompt | model | StrOutputParser()

print(chain.invoke({
    "context": "API keys expire after 30 days on Free, 90 days on Team, and 365 days on Business, "
               "and must be rotated before they expire.",
    "question": "How often do we need to rotate our key? We are on Team.",
}))

print("=== B2: the prompt on its own ===")
msgs = prompt.invoke({"context": "c", "question": "q"})
print(type(msgs).__name__, [m.type for m in msgs.messages])

print("=== C: rag chain ===")
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from resolve.kb.search import search_kb

def retrieve(question: str) -> str:
    hits = search_kb(question, k=4)
    return "\n\n".join(f"[{h['doc_id']}] {h['text']}" for h in hits)

rag_chain = (
    {"context": RunnableLambda(retrieve), "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)
print(rag_chain.invoke("How do I rotate an API key?"))

print("=== C2: batch ===")
answers = rag_chain.batch(["How do I archive a project?", "Is SSO available on the Team plan?"])
for a in answers:
    print("-", a[:160].replace("\n", " "))

print("=== D: @tool ===")
from resolve.tools.kb import search_kb as search_kb_tool
print(search_kb_tool.name)
print(search_kb_tool.args)
print(search_kb_tool.description[:120])
hits = search_kb_tool.invoke({"query": "rotate an API key", "k": 2})
print([(h["doc_id"], h["title"], round(h["score"], 3)) for h in hits])
