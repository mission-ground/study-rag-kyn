from backend.service.rag.components.embedding.embedder import Embedder
from backend.service.rag.ingestion.chunker import Chunker
from backend.service.rag.components.vectorstore.faiss.vector_store import VectorStore
from backend.service.rag.ingestion.document_loader import PdfLoader
import json
import os


#ingestion = 데이터를 시스템에 넣는 과정
class IndexBuilder:

    def __init__(self,index_path="backend/data/embeddings/faiss_index"):

        self.index_path = index_path
        self.docstore_path = f"{index_path}/docstore.json"
        
        self.loader = PdfLoader()
        self.embedder = Embedder()          # 1. 처리할 문서를 임베딩 처리하기 위해 모델을 로딩한다.
        self.chunker = Chunker()            # 2. load 된 데이터를 청킹할 청커를 준비한다.
        self.vector_store = VectorStore()   # 3. 처리된 문서 데이터를 저장할 벡터DB 준비한다.

    # 파일 있으면 로드 or 새로만들기
    def get_or_build(self):
        if  self._exists() and self.vector_store.is_index_valid():
            print("🔥 load 실행")
            return self._load()
        print("🔥 build 실행")
        return self._build()
    
    def _exists(self):
        return (
            os.path.exists(self.index_path) and
            os.path.exists(self.docstore_path)
        )

    # 실제로 처리되는 곳
    def _build(self):

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
        # docstore.json을 저장
        os.makedirs(self.index_path, exist_ok=True)

        with open(self.docstore_path, "w", encoding="utf-8") as f:
            json.dump(docstore, f, ensure_ascii=False)

        return self._build_result(docstore, embeddings, child_contents)
        
    def _load(self):
        # 인덱스, 메타데이타, ids, 청크 로드
        vector_store = VectorStore.load_local(self.index_path)

        # docstore.json을 로드
        with open(self.docstore_path, "r", encoding="utf-8") as f:
            docstore = json.load(f)

        return self._build_result(
            docstore=docstore,
            embeddings=vector_store.index,
            child_contents=vector_store.documents,
            vector_store=vector_store
        )
    
    def _build_result(self, docstore, embeddings, child_contents, vector_store=None):

        return { # todo : embedder, vector_store, docstore만 해도 되겠다
            "embedder": self.embedder,
            "vector_store": vector_store or self.vector_store,
            "docstore": docstore,
            "embeddings": embeddings,
            "child_contents": child_contents
        }