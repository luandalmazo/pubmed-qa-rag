from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

def build_qa_chain(pdf_path="article/example.pdf"):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model="llama3:8b")

    template = """
    Você é um assistente que responde estritamente com base no conteúdo do documento fornecido.
    Se a resposta não estiver no texto, diga: "Não sei com base no PDF fornecido."

    Contexto:
    {context}

    Pergunta: {question}
    """
    QA_PROMPT = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": QA_PROMPT}
    )

    return qa_chain