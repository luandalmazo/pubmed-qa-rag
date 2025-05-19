from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOllama
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate


def save_faiss_index(pdf_path, faiss_folder):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100) 
    docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(faiss_folder)


def build_qa_chain_from_saved_index(faiss_folder):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
    db = FAISS.load_local(faiss_folder, embeddings, allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever(search_kwargs={"k": 4})

    llm = ChatOllama(model="llama3:8b")

    template = """
    You are an assistant that responds only based on the content of the provided document.

    If the answer is not explicitly present in the text, respond exactly with:
    
    "Can't find the answer in the document."

    Do not include suggestions, polite comments, or generic messages (e.g., "Let me know if you want more information").
    Do not cite the question in your answer.
    Do not try to guess or invent information. Return ONLY the answer to the given question.

    Context:
    {context}

    Question: {question}
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
