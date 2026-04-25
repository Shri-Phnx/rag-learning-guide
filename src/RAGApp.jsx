import { useState } from "react";

const C = {
  blue:   "#38bdf8", violet: "#a78bfa", green:  "#34d399",
  amber:  "#fbbf24", pink:   "#f472b6", bg:     "#0d1117",
  panel:  "#161b22", border: "#30363d", muted:  "#8b949e", text: "#e6edf3",
};

function hl(code) {
  return code.split("\n").map((line, li) => {
    const tokens = [];
    let i = 0;
    while (i < line.length) {
      if (line[i] === "#") {
        tokens.push(<span key={i} style={{color:"#6e7681",fontStyle:"italic"}}>{line.slice(i)}</span>);
        i = line.length; continue;
      }
      if (line[i] === '"' || line[i] === "'") {
        const q = line[i]; let j = i + 1;
        while (j < line.length && line[j] !== q) j++;
        tokens.push(<span key={i} style={{color:"#a5d6ff"}}>{line.slice(i,j+1)}</span>);
        i = j + 1; continue;
      }
      const kw = line.slice(i).match(/^(from|import|def|class|return|if|elif|else|for|in|print|with|as|True|False|None|not|and|or)\b/);
      if (kw) { tokens.push(<span key={i} style={{color:C.violet}}>{kw[0]}</span>); i+=kw[0].length; continue; }
      const bi = line.slice(i).match(/^(str|int|list|dict|bool|len|range|f)\b/);
      if (bi) { tokens.push(<span key={i} style={{color:C.blue}}>{bi[0]}</span>); i+=bi[0].length; continue; }
      const nm = line.slice(i).match(/^[0-9]+/);
      if (nm) { tokens.push(<span key={i} style={{color:C.amber}}>{nm[0]}</span>); i+=nm[0].length; continue; }
      tokens.push(<span key={i}>{line[i]}</span>); i++;
    }
    return <div key={li} style={{minHeight:"1.5em"}}>{tokens.length ? tokens : <span>&nbsp;</span>}</div>;
  });
}

const CODE = {
  loader:`from langchain.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.document_loaders.csv_loader import CSVLoader

# ── Load from PDF ─────────────────────────────────────
loader = PyPDFLoader("docs/knowledge_base.pdf")
docs = loader.load()

# ── Load from Web / URL ───────────────────────────────
web_loader = WebBaseLoader("https://example.com/faq")
web_docs = web_loader.load()

# ── Load from CSV ─────────────────────────────────────
csv_loader = CSVLoader("data/products.csv")
csv_docs = csv_loader.load()

print(f"Loaded {len(docs)} document(s)")
# >> Loaded 12 document(s)`,

  splitter:`from langchain.text_splitter import RecursiveCharacterTextSplitter

# Choose chunk size based on your context window:
# 8K, 4K, 128K, 2K, 1M, 2M tokens supported by modern LLMs

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,       # Characters per chunk
    chunk_overlap=200,     # Overlap preserves context at boundaries
    separators=[
        "\\n\\n",
        "\\n",
        ".",
        " ",
    ]
)

chunks = splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks")
# >> Split into 148 chunks`,

  embedding:`from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# Option 1: OpenAI Ada-002 (cloud, high quality, 1536 dims)
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key="sk-..."
)

# Option 2: HuggingFace BGE (free, local, 1024 dims)
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5"
)

# Test: embed a single query
vector = embeddings.embed_query("Why is the sky blue?")
print(f"Vector dimensions: {len(vector)}")
# >> Vector dimensions: 1536`,

  vectorstore:`from langchain.vectorstores import Chroma, FAISS
from langchain.vectorstores.pgvector import PGVector

# Option 1: FAISS — fast, in-memory
db = FAISS.from_documents(chunks, embeddings)
db.save_local("faiss_index")
db = FAISS.load_local("faiss_index", embeddings)

# Option 2: Chroma — persistent local store
db = Chroma.from_documents(
    chunks, embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_knowledge"
)

# Option 3: PgVector — production Postgres
CONNECTION = "postgresql+psycopg2://user:pass@localhost/ragdb"
db = PGVector.from_documents(
    chunks, embeddings,
    connection_string=CONNECTION,
    collection_name="rag_knowledge"
)`,

  retriever:`# Build retriever from your vector store
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 4,
        "score_threshold": 0.7
    }
)

query = "Why is the sky blue?"
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content[:150])
    print(f"Source: {doc.metadata.get('source', 'unknown')}")
    print()`,

  augment:`from langchain.prompts import PromptTemplate

# THE FORMULA:  SP  +  R  +  UQ  →  LLM  →  G
# SP = System Prompt
# R  = Retrieved chunks
# UQ = User Query
# G  = Generated response

template = """You are an expert assistant. Answer ONLY using the
context provided. If not found, say "I don't know."

--- RETRIEVED CONTEXT (R) ---
{context}
--- END CONTEXT ---

USER QUESTION (UQ): {question}

ANSWER:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)`,

  llm:`from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

llm = ChatOpenAI(
    model_name="gpt-4-turbo",
    temperature=0,
    max_tokens=1024,
    openai_api_key="sk-..."
)

chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt, "verbose": True},
    return_source_documents=True
)

result = chain({"query": "Why is the sky blue?"})
print(result["result"])
print(result["source_documents"])`,

  full:`import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

st.set_page_config(page_title="RAG Assistant", page_icon="🧠")
st.title("🧠 RAG-Powered Assistant")

@st.cache_resource
def build_pipeline(doc_path: str):
    docs = PyPDFLoader(doc_path).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200
    ).split_documents(docs)
    embed = OpenAIEmbeddings(model="text-embedding-ada-002")
    db = Chroma.from_documents(chunks, embed)
    retriever = db.as_retriever(search_kwargs={"k": 4})
    prompt = PromptTemplate(
        template="Answer using only the context.\\nContext: {context}\\nQuestion: {question}\\nAnswer:",
        input_variables=["context", "question"]
    )
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True
    )

chain = build_pipeline("knowledge_base.pdf")
query = st.text_input("Ask a question:")
if query:
    with st.spinner("Retrieving and generating..."):
        result = chain({"query": query})
        st.write(result["result"])
        with st.expander("📎 Source documents"):
            for doc in result["source_documents"]:
                st.caption(doc.page_content[:200])`,
};

const PIPELINE = [
  {id:1,phase:"INDEX",icon:"📄",abbr:"DL",label:"Document Loader",color:C.blue,tag:"langchain.document_loaders",ck:"loader",desc:"Ingest raw documents from any source — PDF, web, CSV, Notion, databases, APIs.",detail:"Supports: PDF · DOCX · HTML · CSV · JSON · Web · YouTube · Notion · SQL"},
  {id:2,phase:"INDEX",icon:"✂️",abbr:"TS",label:"Text Splitter",color:C.blue,tag:"RecursiveCharacterTextSplitter",ck:"splitter",desc:"Break documents into smaller, overlapping chunks preserving semantic meaning.",detail:"Context windows: 8K · 4K · 128K · 2K · 1M · 2M tokens"},
  {id:3,phase:"INDEX",icon:"🧬",abbr:"EM",label:"Embedding",color:C.violet,tag:"OpenAIEmbeddings / BGE",ck:"embedding",desc:"Convert text chunks into numerical vectors capturing semantic meaning.",detail:"OpenAI Ada-002 (1536d) · BGE-large (1024d) · Cohere · Google Titan"},
  {id:4,phase:"INDEX",icon:"🗄️",abbr:"VS",label:"Vector Store",color:C.violet,tag:"FAISS · Chroma · PgVector",ck:"vectorstore",desc:"Persist and index vectors for fast similarity search at query time.",detail:"FAISS · Chroma · PgVector · Pinecone · Weaviate · Qdrant"},
  {id:5,phase:"QUERY",icon:"🔍",abbr:"R",label:"Retriever",color:C.amber,tag:"similarity / MMR",ck:"retriever",desc:"Embed the user query then find most semantically similar chunks in the store.",detail:"search_type: similarity · MMR (Max Marginal Relevance) · Hybrid"},
  {id:6,phase:"QUERY",icon:"🧩",abbr:"SP+R+UQ",label:"Augment Prompt",color:C.green,tag:"PromptTemplate",ck:"augment",desc:"Combine System Prompt + Retrieved chunks + User Query into a structured prompt.",detail:"SP (system instructions) + R (retrieved context) + UQ (user question)"},
  {id:7,phase:"QUERY",icon:"🤖",abbr:"LLM",label:"LLM Generate",color:C.green,tag:"ChatOpenAI / Claude",ck:"llm",desc:"The LLM receives the augmented prompt and generates a grounded, factual response.",detail:"GPT-4 · Claude 3 · Gemini 1.5 · Mistral · LLaMA 3 via LangChain"},
  {id:8,phase:"QUERY",icon:"✅",abbr:"G",label:"Response",color:C.pink,tag:"Grounded Answer",ck:null,desc:"Factual, grounded answer returned to the user — with source citations.",detail:"No hallucination · Source-backed · Explainable · Citable"},
];

const TECH = [
  {layer:"UI / Frontend",icon:"🖥️",color:C.blue,tools:[
    {name:"Streamlit",use:"Rapid prototyping, internal tools, demos",effort:"Low",r:5,note:"From your whiteboard — best for MVP"},
    {name:"Gradio",use:"ML demos, HuggingFace Spaces",effort:"Low",r:4,note:"Great for model showcasing"},
    {name:"Flask + React",use:"Production REST APIs + custom UI",effort:"High",r:5,note:"POST/GET — from your whiteboard"},
    {name:"FastAPI",use:"High-performance async REST APIs",effort:"Medium",r:5,note:"Recommended over Flask for production"},
  ]},
  {layer:"Orchestration",icon:"⛓️",color:C.violet,tools:[
    {name:"LangChain (LC)",use:"General RAG, agents, chains, tools",effort:"Medium",r:5,note:"From your whiteboard — most popular"},
    {name:"LlamaIndex",use:"Document-heavy RAG, advanced indexing",effort:"Medium",r:5,note:"Better for complex doc retrieval"},
    {name:"Haystack",use:"Enterprise search + RAG pipelines",effort:"High",r:4,note:"Production-grade, modular"},
    {name:"n8n / Make.com",use:"No-code automation workflows",effort:"Low",r:3,note:"Good for automation pipelines"},
  ]},
  {layer:"Embedding",icon:"🧬",color:C.violet,tools:[
    {name:"OpenAI Ada-002",use:"High quality, general purpose",effort:"Low",r:5,note:"1536 dims · ~$0.0001/1K tokens"},
    {name:"HuggingFace BGE",use:"Free, local, multilingual",effort:"Medium",r:4,note:"BAAI/bge-large-en-v1.5"},
    {name:"Cohere Embed",use:"Multilingual, classification tasks",effort:"Low",r:4,note:"768 / 1024 dims"},
    {name:"Google Vertex AI",use:"GCP ecosystem, enterprise",effort:"Medium",r:4,note:"768 dims, managed service"},
  ]},
  {layer:"Vector DB",icon:"🗄️",color:C.amber,tools:[
    {name:"FAISS",use:"Local dev, speed tests, prototyping",effort:"Low",r:4,note:"In-memory, no server needed"},
    {name:"Chroma",use:"Local/cloud, LangChain default",effort:"Low",r:5,note:"Best for getting started quickly"},
    {name:"Pinecone",use:"Fully managed cloud production",effort:"Low",r:5,note:"Scales to billions of vectors"},
    {name:"PgVector",use:"Existing Postgres infrastructure",effort:"Medium",r:4,note:"From your whiteboard — add to existing DB"},
  ]},
  {layer:"LLM",icon:"🤖",color:C.green,tools:[
    {name:"GPT-4 Turbo",use:"Best quality, complex reasoning",effort:"Low",r:5,note:"128K context window"},
    {name:"Claude 3.5 Sonnet",use:"Long context, technical documents",effort:"Low",r:5,note:"200K context window"},
    {name:"Gemini 1.5 Pro",use:"Multimodal, Google ecosystem",effort:"Low",r:4,note:"1M context window"},
    {name:"LLaMA 3 (local)",use:"Private data, zero API costs",effort:"High",r:4,note:"Self-hosted on your QNAP/GPU server"},
  ]},
];

const CONTEXT_CARDS = [
  {icon:"❓",title:"What is RAG?",color:C.blue,body:"Retrieval-Augmented Generation grounds LLM responses in your own data. Instead of relying on training knowledge with a cutoff date, RAG retrieves relevant information from a live knowledge base and feeds it to the LLM — so answers are accurate, current, and explainable."},
  {icon:"🎯",title:"Why Use RAG?",color:C.green,body:"LLMs hallucinate — they generate plausible but wrong answers. RAG fixes this by forcing the model to answer only from retrieved documents. No expensive fine-tuning needed. Just update your knowledge base. Bonus: every answer can cite its source."},
  {icon:"🔑",title:"The Core Insight",color:C.violet,body:"RAG separates KNOWLEDGE from REASONING. The vector store is your knowledge (updated anytime). The LLM is your reasoning engine. Keeping them separate gives you up-to-date domain knowledge plus powerful language understanding in one system."},
  {icon:"⚠️",title:"When NOT to Use RAG",color:C.amber,body:"RAG is not always the answer. If your task needs new reasoning patterns (not just new facts), or consistent tone/brand voice, fine-tuning may be better. RAG also struggles with questions that require synthesising across many documents simultaneously."},
];

const PITFALLS = [
  {icon:"🔪",title:"Chunking too aggressively",fix:"Use chunk_overlap=200 and preserve sentence boundaries with RecursiveCharacterTextSplitter."},
  {icon:"🎯",title:"Retrieving wrong chunks",fix:"Use MMR search to avoid redundant results. Set a similarity score threshold (≥0.7)."},
  {icon:"📏",title:"Context window overflow",fix:"Keep (k × chunk_size) < LLM context limit. With GPT-4 128K, you have plenty of room."},
  {icon:"🌡️",title:"LLM temperature too high",fix:"Set temperature=0 for factual RAG. Higher values cause creative drift away from retrieved facts."},
  {icon:"🏷️",title:"Missing metadata on chunks",fix:"Always attach source, page number, date to chunks — enables citation and debugging."},
  {icon:"📦",title:"No re-ranking step",fix:"Add a cross-encoder re-ranker (Cohere Rerank) after initial retrieval for higher precision."},
];

const USECASES = [
  {icon:"📚",title:"Enterprise Knowledge Base",eg:"HR policy Q&A, internal SOPs, IT runbooks"},
  {icon:"⚖️",title:"Legal / Compliance Assistant",eg:"Contract review, regulation lookup, ITAM audit responses"},
  {icon:"🏥",title:"Healthcare Information",eg:"Clinical guidelines, drug interactions, patient FAQs"},
  {icon:"💼",title:"Sales & CRM Intelligence",eg:"Personalised pitch generation, competitor analysis"},
  {icon:"🛠️",title:"IT Support Bot",eg:"Codebase Q&A, incident runbooks, ServiceNow automation"},
  {icon:"🎓",title:"EdTech / eLearning",eg:"Course content Q&A, personalised study assistant"},
];

const Chip = ({label,color,small}) => (
  <span style={{fontSize:small?"9px":"10px",fontFamily:"monospace",letterSpacing:"1px",fontWeight:700,
    color,background:color+"18",border:`1px solid ${color}40`,borderRadius:"4px",padding:small?"1px 5px":"2px 7px",whiteSpace:"nowrap"}}>
    {label}
  </span>
);

function StageCard({stage,active,onClick}) {
  return (
    <div onClick={()=>onClick(stage.id)} style={{cursor:"pointer",borderRadius:"10px",padding:"12px",
      background:active?`${stage.color}14`:C.panel,border:`1px solid ${active?stage.color:C.border}`,
      boxShadow:active?`0 0 18px ${stage.color}25`:"none",transition:"all 0.2s",
      display:"flex",flexDirection:"column",gap:"7px",minWidth:0,flex:1}}>
      <div style={{display:"flex",justifyContent:"space-between",alignItems:"flex-start"}}>
        <Chip label={stage.phase} color={stage.phase==="INDEX"?C.blue:C.amber} small />
        <span style={{fontSize:"15px"}}>{stage.icon}</span>
      </div>
      <div style={{fontFamily:"monospace",fontSize:"16px",fontWeight:800,letterSpacing:"-0.5px",lineHeight:1,
        color:active?stage.color:C.muted,transition:"color 0.2s"}}>{stage.abbr}</div>
      <div style={{fontSize:"11px",fontWeight:600,color:C.text,lineHeight:1.3}}>{stage.label}</div>
      <Chip label={stage.tag} color={stage.color} small />
    </div>
  );
}

function Arr({color="#ffffff20"}) {
  return (
    <svg width="18" height="12" style={{flexShrink:0}} viewBox="0 0 18 12">
      <line x1="0" y1="6" x2="12" y2="6" stroke={color} strokeWidth="1.5"/>
      <polygon points="12,3 18,6 12,9" fill={color}/>
    </svg>
  );
}

function PipelineTab() {
  const [activeId,setActiveId] = useState(1);
  const toggle = id => setActiveId(activeId===id?null:id);
  const active = PIPELINE.find(s=>s.id===activeId);
  const idx = PIPELINE.filter(s=>s.phase==="INDEX");
  const qry = PIPELINE.filter(s=>s.phase==="QUERY");

  const Row = ({stages,phaseColor,phaseLabel}) => (
    <div style={{marginBottom:"16px"}}>
      <div style={{display:"flex",alignItems:"center",gap:"7px",marginBottom:"8px"}}>
        <div style={{width:"7px",height:"7px",borderRadius:"50%",background:phaseColor}}/>
        <span style={{fontSize:"9px",fontWeight:700,letterSpacing:"2px",color:phaseColor,fontFamily:"monospace"}}>{phaseLabel}</span>
      </div>
      <div style={{display:"flex",alignItems:"stretch",gap:"5px"}}>
        {stages.map((s,i)=>(
          <div key={s.id} style={{display:"flex",alignItems:"center",gap:"5px",flex:1,minWidth:0}}>
            <StageCard stage={s} active={activeId===s.id} onClick={toggle}/>
            {i<stages.length-1&&<Arr color={phaseColor+"70"}/>}
          </div>
        ))}
      </div>
    </div>
  );

  return (
    <div>
      <Row stages={idx} phaseColor={C.blue} phaseLabel="PHASE 1 — INDEXING (OFFLINE / ONE-TIME)"/>
      <div style={{display:"flex",justifyContent:"center",color:C.violet,fontSize:"9px",fontFamily:"monospace",letterSpacing:"1px",margin:"4px 0"}}>↓ QUERY TIME (PER USER REQUEST)</div>
      <Row stages={qry} phaseColor={C.amber} phaseLabel="PHASE 2 — QUERY & RETRIEVAL (ONLINE / REAL-TIME)"/>
      <div style={{borderRadius:"12px",padding:"16px",marginBottom:"16px",minHeight:"80px",transition:"all 0.25s",background:active?`${active.color}0d`:C.panel,border:`1px solid ${active?active.color+"50":C.border}`}}>
        {active?(<div style={{display:"flex",flexDirection:"column",gap:"8px"}}><div style={{display:"flex",alignItems:"center",gap:"10px"}}><span style={{fontSize:"20px"}}>{active.icon}</span><div><div style={{fontWeight:800,fontSize:"15px",color:active.color}}>{active.label} — {active.abbr}</div><Chip label={active.phase} color={active.phase==="INDEX"?C.blue:C.amber} small/></div></div><p style={{margin:0,color:C.text,fontSize:"13px",lineHeight:1.6}}>{active.desc}</p><div style={{fontFamily:"monospace",fontSize:"11px",color:active.color,background:active.color+"10",border:`1px solid ${active.color}30`,borderRadius:"6px",padding:"7px 10px"}}>{active.detail}</div></div>):(<div style={{color:C.muted,fontSize:"12px",textAlign:"center",paddingTop:"18px"}}>↑ Tap any stage card to see details</div>)}
      </div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"12px",padding:"14px",marginBottom:"14px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1.5px",marginBottom:"12px"}}>🧩 AUGMENTED PROMPT FORMULA</div>
        <div style={{display:"flex",flexWrap:"wrap",gap:"7px",alignItems:"center",justifyContent:"center"}}>
          {[{l:"SP",d:"System Prompt",c:C.violet},{l:"+",c:C.muted},{l:"R",d:"Retrieved Chunks",c:C.blue},{l:"+",c:C.muted},{l:"UQ",d:"User Query",c:C.amber},{l:"→",c:C.muted},{l:"LLM",d:"Processes",c:C.green},{l:"→",c:C.muted},{l:"G",d:"Grounded Answer",c:C.pink}].map((t,i)=>t.d?(<div key={i} style={{textAlign:"center"}}><div style={{background:t.c+"20",border:`1px solid ${t.c}40`,borderRadius:"6px",padding:"5px 11px",fontFamily:"monospace",fontWeight:800,fontSize:"14px",color:t.c}}>{t.l}</div><div style={{fontSize:"9px",color:C.muted,marginTop:"3px"}}>{t.d}</div></div>):(<span key={i} style={{fontFamily:"monospace",fontSize:"17px",color:t.c,fontWeight:300}}>{t.l}</span>))}
        </div>
      </div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"10px",padding:"12px",display:"flex",flexWrap:"wrap",gap:"6px",alignItems:"center"}}>
        <span style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1px",marginRight:"4px"}}>CHUNK / CONTEXT SIZES:</span>
        {["8K","4K","128K","2K","1M","2M"].map(s=><Chip key={s} label={s} color={C.violet} small/>)}
      </div>
    </div>
  );
}

function CodeTab() {
  const [sel,setSel] = useState(PIPELINE[0]);
  const [copied,setCopied] = useState(false);
  const code = sel.ck?CODE[sel.ck]:CODE.full;
  const copy=()=>{ navigator.clipboard?.writeText(code); setCopied(true); setTimeout(()=>setCopied(false),2000); };
  const all = [...PIPELINE,{id:99,phase:"FULL",icon:"🚀",abbr:"ALL",label:"Full Pipeline",color:C.pink,tag:"End-to-End",ck:"full"}];
  return (
    <div style={{display:"flex",flexDirection:"column",gap:"14px"}}>
      <div style={{display:"grid",gridTemplateColumns:"repeat(3, 1fr)",gap:"5px"}}>
        {all.map(s=>(<div key={s.id} onClick={()=>setSel(s)} style={{cursor:"pointer",borderRadius:"8px",padding:"9px 10px",background:sel.id===s.id?`${s.color}20`:C.panel,border:`1px solid ${sel.id===s.id?s.color:C.border}`,display:"flex",alignItems:"center",gap:"6px",transition:"all 0.2s"}}><span style={{fontSize:"13px"}}>{s.icon}</span><div style={{minWidth:0}}><div style={{fontFamily:"monospace",fontWeight:700,fontSize:"11px",color:sel.id===s.id?s.color:C.text,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.abbr}</div><div style={{fontSize:"9px",color:C.muted,overflow:"hidden",textOverflow:"ellipsis",whiteSpace:"nowrap"}}>{s.label}</div></div></div>))}
      </div>
      <div style={{borderRadius:"12px",overflow:"hidden",border:`1px solid ${C.border}`}}>
        <div style={{background:"#1c2128",padding:"9px 14px",display:"flex",alignItems:"center",justifyContent:"space-between",borderBottom:`1px solid ${C.border}`}}>
          <div style={{display:"flex",alignItems:"center",gap:"8px"}}><div style={{display:"flex",gap:"5px"}}>{["#ff5f57","#febc2e","#28c840"].map(c=>(<div key={c} style={{width:"9px",height:"9px",borderRadius:"50%",background:c}}/>))}</div><span style={{fontFamily:"monospace",fontSize:"11px",color:C.muted}}>{sel.label.toLowerCase().replace(/ /g,"_")}.py</span></div>
          <div style={{display:"flex",alignItems:"center",gap:"7px"}}><Chip label={sel.tag||"Python"} color={sel.color||C.green} small/><button onClick={copy} style={{background:copied?C.green+"20":"transparent",border:`1px solid ${copied?C.green:C.border}`,borderRadius:"5px",padding:"3px 9px",cursor:"pointer",color:copied?C.green:C.muted,fontSize:"10px",fontFamily:"monospace",transition:"all 0.2s"}}>{copied?"✓ Copied":"Copy"}</button></div>
        </div>
        <div style={{background:"#010409",overflowX:"auto"}}><table style={{width:"100%",borderCollapse:"collapse"}}><tbody>{code.split("\n").map((line,i)=>(<tr key={i} style={{lineHeight:"1.55"}}><td style={{width:"38px",textAlign:"right",padding:"0 10px",fontFamily:"monospace",fontSize:"11px",color:"#3d444d",userSelect:"none",borderRight:`1px solid ${C.border}`}}>{i+1}</td><td style={{padding:"0 14px",fontFamily:"monospace",fontSize:"12px",color:C.text,whiteSpace:"pre"}}>{hl(line)}</td></tr>))}</tbody></table></div>
      </div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"10px",padding:"13px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,marginBottom:"9px",letterSpacing:"1px"}}>📦 INSTALL DEPENDENCIES</div>
        <div style={{background:"#010409",borderRadius:"8px",padding:"10px 13px",fontFamily:"monospace",fontSize:"11px",color:C.green,lineHeight:"1.9"}}>{["pip install langchain openai chromadb faiss-cpu streamlit","pip install langchain-community tiktoken sentence-transformers","pip install flask flask-cors python-dotenv pypdf"].map((cmd,i)=>(<div key={i}><span style={{color:C.muted}}>$ </span>{cmd}</div>))}</div>
      </div>
    </div>
  );
}

function TechTab() {
  const [lay,setLay] = useState(0);
  const L = TECH[lay];
  return (
    <div style={{display:"flex",flexDirection:"column",gap:"14px"}}>
      <div style={{display:"flex",flexWrap:"wrap",gap:"5px"}}>{TECH.map((l,i)=>(<button key={i} onClick={()=>setLay(i)} style={{background:lay===i?`${l.color}20`:"transparent",border:`1px solid ${lay===i?l.color:C.border}`,borderRadius:"6px",padding:"6px 12px",cursor:"pointer",color:lay===i?l.color:C.muted,fontSize:"11px",fontFamily:"monospace",fontWeight:600,transition:"all 0.2s",display:"flex",alignItems:"center",gap:"5px"}}><span>{l.icon}</span><span style={{fontSize:"10px"}}>{l.layer}</span></button>))}</div>
      <div style={{display:"flex",flexDirection:"column",gap:"8px"}}>{L.tools.map((t,i)=>(<div key={i} style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"10px",padding:"13px",display:"flex",flexDirection:"column",gap:"7px"}}><div style={{display:"flex",justifyContent:"space-between",alignItems:"center",flexWrap:"wrap",gap:"6px"}}><div style={{fontWeight:700,fontSize:"14px",color:C.text}}>{t.name}</div><div style={{display:"flex",gap:"3px"}}>{[...Array(5)].map((_,s)=>(<div key={s} style={{width:"7px",height:"7px",borderRadius:"50%",background:s<t.r?L.color:C.border}}/>))}</div></div><div style={{fontSize:"12px",color:C.muted,lineHeight:1.5}}>{t.use}</div><div style={{display:"flex",justifyContent:"space-between",flexWrap:"wrap",gap:"5px",alignItems:"center"}}><Chip label={`EFFORT: ${t.effort}`} color={t.effort==="Low"?C.green:t.effort==="Medium"?C.amber:C.pink} small/><span style={{fontSize:"11px",color:L.color,fontFamily:"monospace"}}>💡 {t.note}</span></div></div>))}</div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"12px",padding:"14px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1.5px",marginBottom:"12px"}}>⚖️ RAG vs FINE-TUNING</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px"}}>{[{label:"Use RAG when…",color:C.green,points:["Knowledge changes frequently","Need to cite sources","Budget is limited","Quick prototype needed","Data is private / proprietary","Explainability is required"]},{label:"Use Fine-tuning when…",color:C.amber,points:["Need new reasoning patterns","Consistent tone / brand voice","Static curated dataset","Latency matters (no retrieval)","Very specialised task","You have GPU + large dataset"]}].map((col,i)=>(<div key={i} style={{background:col.color+"0a",border:`1px solid ${col.color}30`,borderRadius:"8px",padding:"11px"}}><div style={{fontWeight:700,fontSize:"12px",color:col.color,marginBottom:"9px"}}>{col.label}</div>{col.points.map((p,j)=>(<div key={j} style={{display:"flex",gap:"5px",marginBottom:"5px"}}><span style={{color:col.color,fontSize:"10px",marginTop:"1px",flexShrink:0}}>✓</span><span style={{fontSize:"11px",color:C.text,lineHeight:1.4}}>{p}</span></div>))}</div>))}</div>
      </div>
    </div>
  );
}

function ContextTab() {
  return (
    <div style={{display:"flex",flexDirection:"column",gap:"14px"}}>
      <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"9px"}}>{CONTEXT_CARDS.map((c,i)=>(<div key={i} style={{background:C.panel,border:`1px solid ${c.color}30`,borderRadius:"12px",padding:"15px",display:"flex",flexDirection:"column",gap:"9px"}}><div style={{display:"flex",alignItems:"center",gap:"8px"}}><span style={{fontSize:"18px"}}>{c.icon}</span><div style={{fontWeight:800,fontSize:"13px",color:c.color}}>{c.title}</div></div><p style={{margin:0,fontSize:"12px",color:C.muted,lineHeight:1.7}}>{c.body}</p></div>))}</div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"12px",padding:"14px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1.5px",marginBottom:"12px"}}>📊 RAGAS EVALUATION FRAMEWORK</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"8px",marginBottom:"10px"}}>{[{m:"Faithfulness",d:"Does answer stay within retrieved context?",c:C.green},{m:"Answer Relevancy",d:"How relevant is the answer to the question?",c:C.blue},{m:"Context Recall",d:"Were all relevant chunks retrieved?",c:C.violet},{m:"Context Precision",d:"Were retrieved chunks actually useful?",c:C.amber}].map((m,i)=>(<div key={i} style={{background:m.c+"0d",border:`1px solid ${m.c}30`,borderRadius:"8px",padding:"10px"}}><div style={{fontWeight:700,fontSize:"12px",color:m.c,marginBottom:"4px"}}>{m.m}</div><div style={{fontSize:"11px",color:C.muted}}>{m.d}</div></div>))}</div>
        <div style={{background:"#010409",borderRadius:"6px",padding:"7px 11px",fontFamily:"monospace",fontSize:"11px",color:C.green}}>pip install ragas <span style={{color:C.muted}}>→</span> from ragas import evaluate</div>
      </div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"12px",padding:"14px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1.5px",marginBottom:"12px"}}>⚠️ COMMON PITFALLS & FIXES</div>
        <div style={{display:"flex",flexDirection:"column",gap:"7px"}}>{PITFALLS.map((p,i)=>(<div key={i} style={{display:"flex",gap:"9px",alignItems:"flex-start",padding:"9px 10px",background:"#010409",borderRadius:"8px"}}><span style={{fontSize:"13px",marginTop:"1px"}}>{p.icon}</span><div><div style={{fontWeight:700,fontSize:"12px",color:C.pink,marginBottom:"3px"}}>{p.title}</div><div style={{fontSize:"11px",color:C.muted,lineHeight:1.5}}><span style={{color:C.green}}>Fix: </span>{p.fix}</div></div></div>))}</div>
      </div>
      <div style={{background:C.panel,border:`1px solid ${C.border}`,borderRadius:"12px",padding:"14px"}}>
        <div style={{fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1.5px",marginBottom:"12px"}}>🚀 REAL-WORLD USE CASES</div>
        <div style={{display:"grid",gridTemplateColumns:"1fr 1fr",gap:"7px"}}>{USECASES.map((u,i)=>(<div key={i} style={{display:"flex",gap:"8px",padding:"9px",background:"#010409",borderRadius:"8px",alignItems:"flex-start"}}><span style={{fontSize:"15px"}}>{u.icon}</span><div><div style={{fontWeight:700,fontSize:"12px",color:C.text,marginBottom:"3px"}}>{u.title}</div><div style={{fontSize:"10px",color:C.muted}}>{u.eg}</div></div></div>))}</div>
      </div>
    </div>
  );
}

const TABS = [
  {id:"pipeline",label:"🗺️ Pipeline",comp:PipelineTab},
  {id:"context",label:"📖 Context",comp:ContextTab},
  {id:"code",label:"💻 Code",comp:CodeTab},
  {id:"stack",label:"⚡ Tech Stack",comp:TechTab},
];

export default function RAGApp() {
  const [tab,setTab] = useState("pipeline");
  const Active = TABS.find(t=>t.id===tab).comp;
  return (
    <div style={{background:C.bg,minHeight:"100vh",color:C.text,fontFamily:"'Segoe UI',system-ui,sans-serif"}}>
      <div style={{background:C.panel,borderBottom:`1px solid ${C.border}`,padding:"18px 18px 0",position:"sticky",top:0,zIndex:10}}>
        <div style={{maxWidth:"820px",margin:"0 auto"}}>
          <div style={{fontFamily:"monospace",fontSize:"9px",letterSpacing:"3px",color:C.violet,marginBottom:"4px"}}>AI ARCHITECTURE · LEARNING GUIDE</div>
          <h1 style={{margin:"0 0 14px",fontSize:"clamp(17px,4vw,24px)",fontWeight:900,lineHeight:1.1,background:`linear-gradient(90deg,${C.blue},${C.violet},${C.green})`,WebkitBackgroundClip:"text",WebkitTextFillColor:"transparent",letterSpacing:"-0.5px"}}>RAG — Retrieval-Augmented Generation</h1>
          <div style={{display:"flex",gap:"2px"}}>{TABS.map(t=>(<button key={t.id} onClick={()=>setTab(t.id)} style={{background:tab===t.id?C.bg:"transparent",border:"none",borderTop:tab===t.id?`2px solid ${C.violet}`:"2px solid transparent",padding:"8px 13px",cursor:"pointer",color:tab===t.id?C.text:C.muted,fontSize:"12px",fontWeight:600,transition:"all 0.2s",marginBottom:tab===t.id?"-1px":0}}>{t.label}</button>))}</div>
        </div>
      </div>
      <div style={{maxWidth:"820px",margin:"0 auto",padding:"18px"}}><Active/></div>
      <div style={{borderTop:`1px solid ${C.border}`,padding:"12px",textAlign:"center",fontFamily:"monospace",fontSize:"9px",color:C.muted,letterSpacing:"1px"}}>MY UNDERSTANDING — HOW RAG WORKS & HOW TO IMPLEMENT IT · BUILT FROM WHITEBOARD NOTES</div>
    </div>
  );
}