from backend.service.rag.components.embedding.embedder import Embedder
from backend.service.rag.ingestion.chunker import Chunker
from backend.service.rag.components.vectorstore.faiss.vector_store import VectorStore
from backend.service.rag.ingestion.loaders import PdfLoader
import json

#ingestion = 데이터를 시스템에 넣는 과정
class IndexBuilder:

    def __init__(self):
        
        self.loader = PdfLoader()
        self.embedder = Embedder()          # 1. 처리할 문서를 임베딩 처리하기 위해 모델을 로딩한다.
        self.chunker = Chunker()            # 2. load 된 데이터를 청킹할 청커를 준비한다.
        self.vector_store = VectorStore()   # 3. 처리된 문서 데이터를 저장할 벡터DB 준비한다.

    # 실제로 처리되는 곳
    def build(self):

        # loader = PdfLoader()

        documents = self.loader.load("backend/data/raw/pdf/북브리프_돈의심리학.pdf")

        docstore, child_ids, child_contents, child_metadatas = \
            self.chunker.parent_child_split(documents)

        embeddings = self.embedder.embed(child_contents)

        self.vector_store.add_documents(
            vectors=embeddings,
            docs=child_contents,
            doc_ids=child_ids,
            metadatas=child_metadatas
        )

        with open("docstore.json", "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False)

        return self.embedder, self.vector_store, docstore, embeddings, child_contents
    
    # 추후 Loader를 따로 파일 만들어야 겠다.
    def load_documents(self):

        docs = []

        
        with open("data/documents.txt", "r", encoding="utf-8") as f:

            for line in f:
                docs.append(line.strip())

        return docs
    
    def chunk_documents(self, documents):

        chunks = []
        for doc in documents:

            doc_chunks = self.chunker.split(doc)
            chunks.extend(doc_chunks)
        return chunks
    
