#component = 시스템을 구성하는 하나의 부품
# 텍스트를 벡터로 변경하는 클래스
import time
from typing import List

class Embedder:

    # 생성자
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        t1 = time.time()
        from sentence_transformers import SentenceTransformer
        t2 = time.time()
        print(f" 임베딩 모델 임포트 시간: {t2 - t1:.2f}초")

        start = time.time()
        self.model = SentenceTransformer(model_name)
        end = time.time()
        print(f" 모델 로딩 시간: {end - start:.2f}초")


    # 임베딩
    def embed(self, documents: List[str]):
        
        start = time.time()

        result = self.model.encode(documents, normalize_embeddings=True)
        end = time.time()

        
        print(f" 임베딩 시간: {end - start:.2f}초 (문서 수: {len(documents)})")

        return result

        # return self.model.encode(documents, normalize_embeddings=True)
    


