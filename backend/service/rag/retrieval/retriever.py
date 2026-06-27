# query → embedding → vector search

# RAG(검색 증강 생성) 시스템에서 질문과 관련된 정보를 
# 데이터베이스에서 찾아는 검색 기술 또는 엔진을 의미한다.
# 사람으로 치면 자료 찾는 사람
class Retriever:

    def __init__(self, embedder, vector_store, docstore, embeddings, child_contents):
        self.embedder = embedder
        self.vector_store = vector_store
        self.docstore = docstore
        # self.embeddings = embeddings
        # self.child_contents = child_contents


    def retrieve(self, query, k=3, window_size=1):

        query_embedding = self.embedder.embed([query])
        results = self.vector_store.search_vector(query_embedding, k=k)

        context_chunks = []
        used_indices = set()

        print("=== 검색 결과 index ===")

        for hit in results:
            hit_idx = hit["metadata"]["index"]
            print(hit["metadata"]["index"])

            # 검색한 청크 전후로 청크 이어 붙이기
            for i in range(hit_idx - window_size, hit_idx + window_size + 1):
                if 0 <= i < len(self.child_contents):

                    if i in used_indices: # 이미 사용한 인덱스면 건너뛰기 (중복 방지)
                        continue

                    if i == hit_idx or abs(i - hit_idx) <= window_size:
                        context_chunks.append((i, self.child_contents[i]))
                        used_indices.add(i)

        context_chunks.sort(key=lambda x: x[0]) # 각 요소의 첫 번째 값 기준으로 정렬

    
        # parent 청크 fallback
        parent_ids = set()
        if not context_chunks and results:
            for hit in results[:3]:
                pid = hit["metadata"]["parent_id"]

                if pid in parent_ids:
                  continue
                parent_ids.add(pid)
                context_chunks.append((pid, self.docstore[pid]))

        print("=== 최종 context index ===")
        print([i for i, _ in context_chunks]) # 안 쓰는 변수 = _

        return [c[1] for c in context_chunks]