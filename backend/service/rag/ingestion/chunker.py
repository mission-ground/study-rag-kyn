from transformers import AutoTokenizer
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import kss


class Chunker:

    def __init__(
        self,
        model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        chunk_size=265,
        overlap=50
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.embeding_model = SentenceTransformer(model_name)
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, text): # 토큰 기준

        tokens = self.tokenizer.encode(text)

        chunks = []

        start = 0

        while start < len(tokens):

            end = start + self.chunk_size

            chunk_tokens = tokens[start:end]

            chunk_text = self.tokenizer.decode(chunk_tokens)

            chunks.append(chunk_text)

            start += self.chunk_size - self.overlap

        return chunks
    


    def parent_child_split(
        self,
        documents,
        parent_chunk_size=1500,
        parent_overlap=200,
        child_chunk_size=600,
        child_overlap=50
    ):

        # Recursive : 문단 기준으로 자르고, 안되면 줄바꿈, 안되면 공백, 최후 → 글자 단위
        # parent_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=parent_chunk_size,
        #     chunk_overlap=parent_overlap
        # )

        child_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            separators = ["\n\n", "\n", ".", "다.", "요."],
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap
        )

        docstore = {}
        child_contents = []
        child_metadatas = []
        child_ids = []

        for doc in documents:

            parents = self.semantic_chunking(doc)
            
            # parent_splitter.split_text(doc)

            for parent in parents:

                parent_id = f"parent_{uuid.uuid4().hex[:8]}"
                docstore[parent_id] = parent

                children = child_splitter.split_text(parent)

                seen = set()
                for child in children:
                    if child in seen: # 중복 chunk 발생X
                        continue
                    seen.add(child)

                    child_id = f"chunk_{uuid.uuid4().hex[:8]}"

                    global_idx = len(child_contents)

                    child_contents.append(child)
                    child_metadatas.append({
                        "parent_id": parent_id,
                        "index": global_idx
                    })
                    child_ids.append(child_id)

        return docstore, child_ids, child_contents, child_metadatas
    



    def semantic_chunking(self, text, threshold=0.3):
        # 1. 문장 단위 분리
        sentences = self.split_sentences(text)

        if not sentences:
            return []

        if len(sentences) == 1:
            return sentences

        # 2. 임베딩
        embeddings = self.embeding_model.encode(
            sentences,
            batch_size=32,
            normalize_embeddings=True) # 벡터 길이 1로 정규화

        chunks = [] # parent chunk 리스트
        current_chunk = [sentences[0]]

        # for i in range(1, len(sentences)):
        #     sim = cosine_similarity(
        #         [embeddings[i-1]], [embeddings[i]]
        #     )[0][0]

        max_length = 1500  # parent 목표 길이
        current_len = 0

        sims = np.sum(embeddings[:-1] * embeddings[1:], axis=1) # 문장 i vs 문장 i+1 유사도 계산

        for i, sim in enumerate(sims):
            next_sent = sentences[i+1]
            current_len += len(next_sent)

            if current_len > max_length:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            elif sim < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_len = 0

            current_chunk.append(next_sent)

        if current_chunk:
            chunks.append(" ".join(current_chunk))


        # print(f"★전체 청크!!!!!!!!!!! {chunks}")
        return chunks
    

    def split_sentences(self, text):
        return kss.split_sentences(text) # kss = 한국어 문장 분리기.. 긴 텍스트 → 문장 리스트로 변환
