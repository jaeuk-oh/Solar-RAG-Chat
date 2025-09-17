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

############# streamlit 배포 시 chromadb와 sqlite 버전 안 맞음 ##################
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
##############################################################################

# .env 파일에서 upstage key 받아오기
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("UPSTAGE_API_KEY")


# 세션 상태 초기화
if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.uploaded_files = []
    st.session_state.vectorstore = None
    st.session_state.rag_chain = None


# 세션 ID 설정
session_id = st.session_state.id
client = None


# 채팅 초기화 함수 정의
def reset_chat() -> None:
    """나눴던 대화와 불러온 문서 초기화하는 함수
    """
    st.session_state.messages = []
    st.session_state.context = None


# 읽어온 PDF 를 보여주는 함수
def display_pdf(file_bytes, filename) -> None:
    """PDF 파일을 받아와서 디스플레이 해주는 함수
    """
    st.markdown(f"### PDF Preview: {filename}")
    base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf" style="height:100vh; width:100%"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


# 여러 PDF 파일 처리 함수
def process_multiple_pdfs(uploaded_files):
    """여러 PDF 파일을 처리하여 벡터 저장소를 만드는 함수
    """
    all_pages = []
    
    for uploaded_file in uploaded_files:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                loader = PyPDFLoader(file_path)
                pages = loader.load_and_split()
                all_pages.extend(pages)
                
        except Exception as e:
            st.error(f"파일 {uploaded_file.name} 처리 중 오류 발생: {e}")
            continue
    
    if all_pages:
        # 모든 페이지를 하나의 벡터스토어로 만들기
        vectorstore = Chroma.from_documents(all_pages, UpstageEmbeddings(model="solar-embedding-1-large"))
        retriever = vectorstore.as_retriever(k=2)
        
        chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))
        
        contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # 이전 대화를 기억하는 리트리버 생성
        history_aware_retriever = create_history_aware_retriever(
            chat, retriever, contextualize_q_prompt
        )
        
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
        
        return vectorstore, rag_chain
    
    return None, None


with st.sidebar:
    st.header(f"Add your documents!")
    
    # 여러 파일 업로드 가능하게 수정
    uploaded_files = st.file_uploader(
        "Choose your `.pdf` files", 
        type="pdf", 
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # 파일이 새로 업로드되었는지 확인
        current_file_names = [f.name for f in uploaded_files]
        previous_file_names = [f['name'] for f in st.session_state.uploaded_files]
        
        if current_file_names != previous_file_names:
            st.write("Indexing your documents ...")
            
            # 새로운 파일들 저장
            st.session_state.uploaded_files = [
                {'name': f.name, 'content': f.getvalue()} 
                for f in uploaded_files
            ]
            
            # 벡터스토어와 RAG 체인 생성
            vectorstore, rag_chain = process_multiple_pdfs(uploaded_files)
            
            if vectorstore and rag_chain:
                st.session_state.vectorstore = vectorstore
                st.session_state.rag_chain = rag_chain
                st.success("Ready to Chat!")
            else:
                st.error("문서 처리 중 오류가 발생했습니다.")
        
        # 업로드된 파일 목록 표시
        st.write("**업로드된 파일들:**")
        for i, file_info in enumerate(st.session_state.uploaded_files):
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"📄 {file_info['name']}")
            with col2:
                if st.button("미리보기", key=f"preview_{i}"):
                    st.session_state.preview_file = i
    
    # 미리보기 표시
    if hasattr(st.session_state, 'preview_file') and st.session_state.preview_file is not None:
        if st.session_state.preview_file < len(st.session_state.uploaded_files):
            file_info = st.session_state.uploaded_files[st.session_state.preview_file]
            display_pdf(file_info['content'], file_info['name'])


# 웹사이트 제목 작성
st.title("Solar RAG Chatbot")

# 업로드된 파일 정보 표시
if st.session_state.uploaded_files:
    st.info(f"현재 {len(st.session_state.uploaded_files)}개의 PDF 파일이 로드되어 있습니다.")

# 메세지 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 기존 메세지 표시
for message in st.session_state.messages:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# 기록하는 대화의 최대 길이를 설정
MAX_MESSAGES_BEFORE_DELETION = 8

# 유저입력 처리
if prompt := st.chat_input("질문을 입력하세요!"):
    # RAG 체인이 준비되어 있는지 확인
    if st.session_state.rag_chain is None:
        st.error("먼저 PDF 파일을 업로드해주세요!")
    else:
        # 이전 대화의 길이 확인
        if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
            del st.session_state.messages[0]
            del st.session_state.messages[0]

        st.session_state.messages.append(
            {"role": "user","content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # AI 의 답변을 받아서 session state에 저장하고, 보여도 줘야함
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""

            result = st.session_state.rag_chain.invoke(
                {'input': prompt, 'chat_history': st.session_state.messages}
            )
            
            with st.expander("불러온 문서"):
                st.write(result['context'])

            for chunk in result['answer'].split(" "):
                full_response += chunk + " "
                message_placeholder.markdown(full_response)
        
        st.session_state.messages.append(
            {"role": "assistant","content": full_response})