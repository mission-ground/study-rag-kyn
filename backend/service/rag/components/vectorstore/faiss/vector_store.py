import faiss
import numpy as np
import os
import json
from backend.core.config import INDEX_PATH, META_PATH, DOCS_PATH


class VectorStore:

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension) #IndexFlatL2 = 유클리드 거리기반 검색
        self.documents = [] # child_contents
        self.metadatas = []
        self.ids = []

        self.index_path = INDEX_PATH
        self.meta_path = META_PATH
        self.docs_path = DOCS_PATH

    # build()시에 실행
    def add_documents(self, vectors, docs, doc_ids, metadatas):
        vectors = np.array(vectors).astype('float32') # FAISS는 내부적으로 C++ 기반이라서 Python list안되고, numpy array만 받는다.
        self.index.add(vectors)

        self.documents.extend(docs)
        self.ids.extend(doc_ids)
        self.metadatas.extend(metadatas)

        self.save_local(self.index_path)
        self.save_meta()
        print(f"총 인덱스 갯수!!! {self.index.ntotal}")
        # 0번 벡터 확인
        # vec = np.zeros((1, 384), dtype='float32')  
        # self.index.reconstruct(0, vec[0]) 

        # print(vec)

    # FAISS 검색
    def search_vector(self, query_vector, k):

        query_vector = np.array(query_vector).astype('float32')
        distances, indices = self.index.search(query_vector, k) # 거리값과 인덱스 번호

        results = []

        for idx, dist in zip(indices[0], distances[0]): # 두 개 리스트를 짝지어서 묶어주는 함수
            if idx == -1: # 결과 없으면
                continue

            results.append({
                "document": self.documents[idx],
                "id": self.ids[idx],
                "metadata": self.metadatas[idx],
                "distance": dist # L2는 작을수록 유사함
            })

        print(f"★총 검색 결과!!! {results}")

        return results

    
    def save_local(self, path):

        os.makedirs(path, exist_ok=True)

        # FAISS index
        faiss.write_index(self.index, f"{path}/index.faiss")

        # metadata
        with open(f"{path}/metadatas.json", "w", encoding="utf-8") as f:
            json.dump(self.metadatas, f, ensure_ascii=False)

        # ids
        with open(f"{path}/ids.json", "w", encoding="utf-8") as f:
            json.dump(self.ids, f, ensure_ascii=False)

        # documents (child_contents)
        with open(f"{path}/chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False)

    @classmethod
    def load_local(cls, path, dimension=384):

        self = cls(dimension)

        self.index = faiss.read_index(f"{path}/index.faiss")

        with open(f"{path}/metadatas.json", "r", encoding="utf-8") as f:
            self.metadatas = json.load(f)

        with open(f"{path}/ids.json", "r", encoding="utf-8") as f:
            self.ids = json.load(f)

        with open(f"{path}/chunks.json", "r", encoding="utf-8") as f:
            self.documents = json.load(f)

        return self
            
    # 모든 파일 중 가장 최신 수정 시간
    def get_docs_last_modified(self):
        latest = 0
        for root, _, files in os.walk(self.index_path):
            for f in files:
                path = os.path.join(root, f)
                latest = max(latest, os.path.getmtime(path)) # 최신 파일 기준 현재 docs 폴더 상태
        print("!! 최신 상태 : ",latest)
        return latest

    # build-load 분기 전에 실행
    # “인덱스 만들 때 기준 시간(과거)” vs “지금 현재 폴더 상태 시간(현재)”
    def is_index_valid(self):
        if not os.path.exists(self.index_path):
            return False

        if not os.path.exists(self.meta_path):
            return False

        with open(self.meta_path, "r") as f:
            meta = json.load(f) # meta.json

        print("!! 기준 시간 : ",meta["last_modified"])
        return self.is_same_time(meta["last_modified"], self.get_docs_last_modified())
    
    # build()시 실행
    # docs 폴더 안 파일들의 마지막 수정 시간 중 가장 최신 값
    def save_meta(self):
        os.makedirs(self.index_path, exist_ok=True)

        meta = {
            "last_modified": self.get_docs_last_modified()
        }

        with open(self.meta_path, "w") as f:
            json.dump(meta, f)

    # 허용 오차(epsilon)
    def is_same_time(self, t1, t2, eps=0.01):  # 10ms 허용
        return abs(t1 - t2) < eps