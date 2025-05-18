from core import build_qa_chain 
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import torch 

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

def get_article_summary(DOI):
    esearch_params = {
        "db": "pubmed",
        "term": f"{DOI}[DOI]",
        "retmode": "xml"
    }

    esearch_response = requests.get(ESEARCH_URL, params=esearch_params)
    esearch_tree = ET.fromstring(esearch_response.text)
    id_elem = esearch_tree.find(".//Id")

    if id_elem is None:
        print("PMID não encontrado para o DOI informado.")
        return

    pmid = id_elem.text
    print(f"PMID encontrado: {pmid}")

    esummary_params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }

    esummary_response = requests.get(ESUMMARY_URL, params=esummary_params)
    esummary_tree = ET.fromstring(esummary_response.text)
    docsum = esummary_tree.find(".//DocSum")

    info = {}
    if docsum is not None:
        for item in docsum.findall("Item"):
            name = item.attrib.get("Name")
            if name in ["Title", "Source", "PubDate", "AuthorList"]:
                info[name] = item.text if name != "AuthorList" else [author.text for author in item.findall("Item")]

    efetch_params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "xml"
    }

    efetch_response = requests.get(EFETCH_URL, params=efetch_params)
    efetch_tree = ET.fromstring(efetch_response.text)
    article = efetch_tree.find(".//PubmedArticle")

    language = article.findtext(".//Language")
    country = article.findtext(".//MedlineJournalInfo/Country")
    publication_type = article.findtext(".//PublicationType")

    afiliacoes = set()
    for affil in article.findall(".//AffiliationInfo/Affiliation"):
        afiliacoes.add(affil.text)

    print("\nInformações do Artigo:")
    print(f"Título do artigo: {info.get('Title', 'N/A')}")
    print(f"Título do Periódico: {info.get('Source', 'N/A')}")
    print(f"Autores: {', '.join(info.get('AuthorList', []))}")
    print(f"Ano de Publicação: {info.get('PubDate', 'N/A')}")
    print(f"Idioma: {language or 'N/A'}")
    print(f"País: {country or 'N/A'}")
    print(f"Tipo de Publicação: {publication_type or 'N/A'}")
    print("Afiliações:")
    for a in afiliacoes:
        print(f" - {a}")

   
if __name__ == "__main__":

    ''' Getting info about the article'''
    #DOI = "10.1038/s41436-018-0299-7"
    #get_article_summary(DOI)

    print("Starting.")
    print("Using GPU:", torch.cuda.is_available())
    chat_history = []
    qa_chain = build_qa_chain("article/article.pdf")

    ''' Reading questions to be answered about the article '''
    df = pd.read_csv("questions.csv")  
    answers = []

    start = time.time()
    for _, row in df.iterrows():
        contextualized_question = f"Na {row['Campo'].lower()}, {row['Pergunta']}"
        result = qa_chain({"question": contextualized_question, "chat_history": chat_history})
        answer = result["answer"]

        print("\n❓ Pergunta:", contextualized_question)
        print("--------------------------------------------------------")
        print("\n💬 Resposta:", answer)
        print("--------------------------------------------------------")

        print("======================================")

        answers.append({
            "Campo": row['Campo'],
            "Pergunta": row['Pergunta'],
            "PerguntaContextualizada": contextualized_question,
            "Resposta": answer,
        })

    end = time.time()

    df_answers = pd.DataFrame(answers)
    df_answers.to_csv("answer.csv", index=False)

    print("Time taken:", end - start)
    print("Answers saved in answer.csv.")
    print("Goodbye..")