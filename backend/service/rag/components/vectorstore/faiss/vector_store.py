import faiss
import numpy as np

class VectorStore:

    def __init__(self, dimension=384):
        self.index = faiss.IndexFlatL2(dimension) #IndexFlatL2 = 유클리드 거리기반 검색
        self.documents = []
        self.metadatas = []
        self.ids = []

    def add_documents(self, vectors, docs, doc_ids, metadatas):
        vectors = np.array(vectors).astype('float32') # FAISS는 내부적으로 C++ 기반이라서 Python list안되고, numpy array만 받는다.
        self.index.add(vectors)

        self.documents.extend(docs)
        self.ids.extend(doc_ids)
        self.metadatas.extend(metadatas)

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