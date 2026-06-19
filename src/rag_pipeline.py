import os
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
import chromadb

class LocalRAGPipeline:
    def __init__(self,db_dir="./chroma_db"):
        self.client=genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.chroma=chromadb.PersistentClient(path=db_dir)
        self.collection=self.chroma.get_or_create_collection("support_kb")

    def embed(self, text):
        response = self.client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        return response.embeddings[0].values

    def load_documents(self,data_dir="data"):
        docs=[]
        for f in os.listdir(data_dir):
            p=os.path.join(data_dir,f)
            if f.endswith((".txt",".md")):
                docs.append((f,open(p,encoding="utf-8").read()))
            elif f.endswith(".pdf"):
                reader=PdfReader(p); txt=""
                for page in reader.pages: txt += (page.extract_text() or "")+"\n"
                docs.append((f,txt))
        return docs

    def ingest_all(self,data_dir="data"):
        splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
        for name,text in self.load_documents(data_dir):
            for i,ch in enumerate(splitter.split_text(text)):
                self.collection.add(ids=[f"{name}_{i}"],documents=[ch],embeddings=[self.embed(ch)],metadatas=[{"source":name}])

    def retrieve(self,query,top_k=3):
        res=self.collection.query(query_embeddings=[self.embed(query)],n_results=top_k)
        out=[]
        for i,d in enumerate(res["documents"][0]):
            score=1-(res["distances"][0][i] if res.get("distances") else 0)
            out.append({"text":d,"source":res["metadatas"][0][i]["source"],"score":score})
        return out
