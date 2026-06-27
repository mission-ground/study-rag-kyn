# main.py
# 역할: 각 클래스를 조립해서 전체 RAG 파이프라인을 실행한다.
# 전체 흐름 : PDFLoader → Chunker → Embedder → Retriever → Generator

import time

t0 = time.time()
print(" 시작")

from backend.service.rag.ingestion.index_builder import IndexBuilder
t1 = time.time()
print(f" IndexBuilder import: {t1 - t0:.2f}초")

from backend.service.rag.rag_pipeline import RAGPipeline
t2 = time.time()
print(f" RAGPipeline import: {t2 - t1:.2f}초")

import gradio as gr
t3 = time.time()
print(f" gradio import: {t3 - t2:.2f}초")


import traceback
import threading

#1. 데이터 로드 → 청킹 → 임베딩 → 벡터DB 저장 → 검색 → 답변 생성

pipeline = None
is_loading = False
build_done = threading.Event()   # 빌드 완료 신호

lock = threading.Lock() # 여러 스레드가 동시에 pipeline 만들지 못하게 막는다.


def preload():
    global pipeline, is_loading

    with lock:
        if pipeline is not None or is_loading:
            return
        is_loading = True

    print(" preload 시작")

    try:

        t0 = time.time()
        builder = IndexBuilder()
        t1 = time.time()
        print(f" 인덱스빌더 초기화 {t1 - t0:.2f}초")

        data = builder.get_or_build()
        t2 = time.time()

        print(f" 빌드 완료 {t2 - t1:.2f}초")
        local_pipeline = RAGPipeline(data["embedder"], data["vector_store"], data["docstore"], data["embeddings"], data["child_contents"])
        t3 = time.time()
        print(f" 파이프 생성 {t3 - t2:.2f}초")
        print(f" 전체 완료 {t3 - t0:.2f}초")

    
        # 로컬에서 만든 걸 전역에 교체
        with lock:
            pipeline = local_pipeline
            is_loading = False
            build_done.set()   # 완료 신호

    
    except Exception as e:
        print(" preload 실패:", e)
        build_done.set() 


    finally:
        with lock:
            is_loading = False
            
# 앱 시작 시 백그라운드 실행
threading.Thread(target=preload, daemon=True).start()

# ============================================================================

# Gradio용 함수
def chat_fn(message, history):

    global pipeline


    if not build_done.is_set():
        print(" build 기다리는 중...")
        build_done.wait()   # 여기서 대기

    #=========================

    try:
        answer = pipeline.ask(message)
    except Exception as e:
        answer = f"에러 발생: {e}"
        traceback.print_exc()

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    
    return history, history, ""  # ← 마지막이 입력창 초기화


# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# RAG 챗봇")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="질문을 입력하세요!!")

    msg.submit(chat_fn, [msg, chatbot], [chatbot, chatbot, msg])

# 실행
demo.launch()