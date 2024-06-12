import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from transformers import pipeline
import pyttsx3

# Initialize the text-to-speech engine
tts_engine = pyttsx3.init()

def text_to_speech(text):
    tts_engine.say(text)
    tts_engine.runAndWait()

def get_pdf_text(pdf_docs):
    pdf_texts = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        pdf_texts[pdf.name] = text
    return pdf_texts

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(chunks):
    embeddings_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    embeddings = embeddings_model.embed_documents(chunks)

    st.write(f"Number of chunks: {len(chunks)}")
    st.write(f"Number of embeddings: {len(embeddings)}")

    if len(embeddings) == 0 or len(chunks) == 0:
        raise ValueError("No chunks or embeddings were generated. Check the input text and model.")

    vector_store = FAISS.from_texts(texts=chunks, embedding=embeddings_model)
    return vector_store, chunks 

def get_conversation_chain(vectorstore, chunks):
    api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
    if not api_token:
        st.error("HUGGINGFACEHUB_API_TOKEN not found. Please set the token in the .env file.")
        return None
    
    llm = HuggingFaceHub(
        repo_id='google/flan-t5-base',
        model_kwargs={"temperature": 0.5, "max_length": 512}, 
        huggingfacehub_api_token=api_token,
    )

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    if "summarize" in user_question.lower():
        if st.session_state.pdf_texts:
            with st.spinner("Summarizing"):
                for pdf_name, pdf_text in st.session_state.pdf_texts.items():
                    summary = summarize_text(pdf_text)
                    st.markdown(f"### Summary of **{pdf_name}**")
                   
        else:
            st.write("No PDF text available to summarize. Please upload and process the PDFs first.")
    elif st.session_state.conversation:
        if st.session_state.mode == 'combined':
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history = response['chat_history']

            for i, message in enumerate(st.session_state.chat_history):
                if i % 2 == 0:
                    st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                    if st.button("🔊", key=f"tts_user_{i}"):
                        text_to_speech(message.content)
                else:
                    st.markdown(f'''
                        <div class="answer-block">
                            <p>{message.content}</p>
                            <button class="speaker-button" onclick="speechSynthesis.speak(new SpeechSynthesisUtterance(`{message.content}`))">🔊</button>
                        </div>
                    ''', unsafe_allow_html=True)
        elif st.session_state.mode == 'separate':
            for pdf_name, conversation in st.session_state.conversations.items():
                st.markdown(f"### Responses from **{pdf_name}**")
                response = conversation({'question': user_question})
                chat_history = response['chat_history']

                for i, message in enumerate(chat_history):
                    if i % 2 == 0:
                        st.markdown(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
                        if st.button("🔊", key=f"tts_user_{pdf_name}_{i}"):
                            text_to_speech(message.content)
                    else:
                        st.markdown(f'''
                            <div class="answer-block">
                                <p>{message.content}</p>
                                <button class="speaker-button" onclick="speechSynthesis.speak(new SpeechSynthesisUtterance(`{message.content}`))">🔊</button>
                            </div>
                        ''', unsafe_allow_html=True)
    else:
        st.write("Conversation chain is not initialized. Please upload and process the PDFs first.")

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary = summarizer(text, max_length=200, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "pdf_texts" not in st.session_state:
        st.session_state.pdf_texts = None
    if "conversations" not in st.session_state:
        st.session_state.conversations = None
    if "mode" not in st.session_state:
        st.session_state.mode = 'combined'

    st.header("Chat with multiple PDFs :books:")

    # Mode selection buttons
    st.markdown('<div class="mode-button-container">', unsafe_allow_html=True)
    if st.button("Separate", key="separate_mode"):
        st.session_state.mode = "separate"
    if st.button("Combined", key="combined_mode"):
        st.session_state.mode = "combined"
    if st.button("ℹ️", key="info_button"):
        st.info("**Separate Mode:** Responses are generated separately for each uploaded PDF.\n\n**Combined Mode:** A single response is generated using the content of all uploaded PDFs combined.")
    st.markdown('</div>', unsafe_allow_html=True)

    user_question = st.text_input("Ask a question about your documents (type 'summarize' to get a summary):")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                pdf_texts = get_pdf_text(pdf_docs)
                st.session_state.pdf_texts = pdf_texts  # Store the raw texts in session state
                st.session_state.conversations = {}

                for pdf_name, pdf_text in pdf_texts.items():
                    text_chunks = get_text_chunks(pdf_text)
                    vectorstore, chunks = get_vectorstore(text_chunks)
                    conversation_chain = get_conversation_chain(vectorstore, chunks)
                    st.session_state.conversations[pdf_name] = conversation_chain

                st.session_state.conversation = get_conversation_chain(
                    *get_vectorstore(" ".join(st.session_state.pdf_texts.values()))
                )

if __name__ == '__main__':
    main()
