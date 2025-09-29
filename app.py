import time
import os
import base64
import uuid
import tempfile
from typing import Dict, List, Any, Optional
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

from langchain_upstage import ChatUpstage
from langchain_core.messages import HumanMessage, SystemMessage

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv
import streamlit as st

# 환경변수에 키 불러오기
load_dotenv()

# 세션 상태 초기화
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {} 

# 세션 ID 설정
session_id = st.session_state.id
client = None

# 채팅 초기화 함수
def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None

# PDF 파일 디스플레이 함수 정의
def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)

# 사이드바 구성
with st.sidebar:
    st.header(f"Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")
    # 파일 업로드 처리
    if uploaded_file:
        print(uploaded_file)
        try:
            file_key = f"{session_id}-{uploaded_file.name}"
            st.write("Indexing your document...")
            # 임시 디렉토리 생성 및 파일 저장
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print("file path:", file_path)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                # PDF 로더 생성 및 문서 분할
                if file_key not in st.session_state.get('file_cache', {}):
                    if os.path.exists(temp_dir):
                        print("temp_dir:", temp_dir)
                        loader = PyPDFLoader(file_path)
                    # 파일 경로 확인 및 에러 처리
                    else:
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()
                    # 페이지 로드 및 벡터 스토어 생성
                    pages = loader.load_and_split()
                    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))

                    # 리트리버 생성
                    retriever = vectorstore.as_retriever(k=2)

                    # 챗봇 생성
                    api_key = os.getenv("UPSTAGE_API_KEY")
                    chat = ChatUpstage(api_key=api_key)

                    # 질문 재구성 프롬프트
                    contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

                    contextualize_q_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", contextualize_q_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    # 히스토리 기반 리트리버 생성
                    history_aware_retriever = create_history_aware_retriever(
                        chat, retriever, contextualize_q_prompt
                    )

                    # 질문 답변 체인 생성
                    qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 질문에 답하기 위해 검색된 내용을 사용하세요. 답을 모르면 모른다고 말하세요. 답변은 세 문장 이내로 간결하게 유지하세요.
                    ## 답변 예시
                    📍답변 내용:
                    📍증거:
                    {context}"""
                    qa_prompt = ChatPromptTemplate.from_messages(
                        [
                            ("system", qa_system_prompt),
                            MessagesPlaceholder("chat_history"),
                            ("human", "{input}"),
                        ]
                    )
                    question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)
                    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
                    
                    # PDF 파일 디스플레이
                    st.success("Ready to Chat!")
                    display_pdf(uploaded_file)
        except Exception as e:
                st.error(f"An error occuered : {e}")
                st.stop()

# 웹사이트 제목 설정
st.title("논문 같이 읽어줄게.💻#️⃣")

if st.button("대화 초기화"):
    reset_chat()

# 파일 입력하고 시도하도록 얼럿 추가
if not uploaded_file:
    st.toast("논문을 입력하셔야 대화를 시작하실 수 있습니다.")
    
# 메세지 초기화
# 시작한 대화가 없으면 빈 값으로 유지를 해라.
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메세지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]): # role = user, assistant
        st.markdown(message["content"])

# 기록하는 대화의 최대갯수 설정
MAX_MESSAGES_BEFORE_DELETION = 12

# 유저 입력 처리
if prompt := st.chat_input("Ask a question!"):
	# 이전 대화의 길이 확인
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        del st.session_state.messages[0]
        del st.session_state.messages[0]
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답처리
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})
        with st.expander("참고 자료"):
            st.write(result["context"])
        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response+ "▌")
        message_placeholder.markdown(full_response)
        
        st.session_state.messages.append(
            {"role": "assistant","content": full_response})