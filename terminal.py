from core import save_faiss_index, build_qa_chain_from_saved_index
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
import torch 
import re
import requests
import xml.etree.ElementTree as ET
import os
import argparse

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def get_article_content_from_pdf_filename(pdf_filename, csv_path='article_info.csv'):
    match = re.match(r'(\d+)\.pdf', pdf_filename)

    if not match:
        print("Nome de arquivo inválido. Deve estar no formato 'ID.pdf', como '322.pdf'.")
        return None

    article_id = int(match.group(1))

    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Arquivo CSV '{csv_path}' não encontrado.")
        return None

    row = df[df['ID'] == article_id]

    if row.empty:
        print(f"ID {article_id} não encontrado na planilha.")
        return None

    return row.iloc[0]['Content']

def extract_pmid_from_text(text):
  
    match = re.search(r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', text)
    if match:
        return match.group(1)
    else:
        print("PMID não encontrado no texto fornecido.")
        return None
    
def get_article_summary_by_pmid(pmid):
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

    affiliations = set()
    for affil in article.findall(".//AffiliationInfo/Affiliation"):
        affiliations.add(affil.text)

    return {
        "Title": info.get("Title", "N/A"),
        "Journal": info.get("Source", "N/A"),
        "Authors": info.get("AuthorList", []),
        "PublicationDate": info.get("PubDate", "N/A"),
        "Language": language or "N/A",
        "Country": country or "N/A",
        "PublicationType": publication_type or "N/A",
        "Affiliations": list(affiliations)
    }

ARTICLE_DIR = './article/'  
INDEX_DIR = './indexes/'

os.makedirs(INDEX_DIR, exist_ok=True)


if __name__ == "__main__":
    print("--------------------------------------------")
    print("Starting.")
    print("Using GPU:", torch.cuda.is_available())
    print("Starting at:", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

    parser = argparse.ArgumentParser(description="Process articles and answer questions.")
    parser.add_argument("--should_save_index", action="store_true", help="If set, FAISS indexes will be saved.")

    args = parser.parse_args()
    should_save_index = args.should_save_index

    print("Should save index?", should_save_index)

    chat_history = []
    df_questions = pd.read_csv("questions.csv")  
    all_answers = []

    if should_save_index:
        print("Ok, let's save the indexes.")
        for article_file in os.listdir(ARTICLE_DIR):

            if not article_file.endswith(".pdf"):
                continue

            pdf_path = os.path.join(ARTICLE_DIR, article_file)
            index_name = article_file.replace(".pdf", "")
            index_path = os.path.join(INDEX_DIR, index_name)

            if not os.path.exists(index_path):
                print(f"Salvando index para {article_file}...")
                save_faiss_index(pdf_path, index_path)
    else: 
        print("Ok, let's load the indexes.")

    print("--------------------------------------------")

    print("Processing articles...")
    for article_file in os.listdir(ARTICLE_DIR):

        pmid = None
        article_data = None

        ''' Only process PDF files '''
        if not article_file.endswith(".pdf"):
            break

        description = get_article_content_from_pdf_filename(article_file)
        pmid = extract_pmid_from_text(description)

        if not pmid:
            continue

        article_data = get_article_summary_by_pmid(pmid)

        if not article_data:
            continue

        index_name = article_file.replace(".pdf", "")
        index_path = os.path.join("indexes", index_name)
        qa_chain = build_qa_chain_from_saved_index(index_path)

        for _, row in df_questions.iterrows():
            contextualized_question = f"According to the {row['Campo'].lower()} of the article, {row['Pergunta']}"
            result = qa_chain({"question": contextualized_question, "chat_history": chat_history})
            answer = result["answer"]

            print("\n❓ Pergunta:", contextualized_question)
            print("--------------------------------------------------------")
            print("\n💬 Resposta:", answer)
            print("--------------------------------------------------------")
            print("======================================")

            all_answers.append({
                "ArticleID": article_file.replace(".pdf", ""),
                "PMID": pmid,
                "Titulo": article_data.get("Title"),
                "Autores": "; ".join(article_data.get("Authors", [])),
                "Afiliações": "; ".join(article_data.get("Affiliations", [])),
                "Linguagem": article_data.get("Language"),
                "País": article_data.get("Country"),
                "Tipo de Publicação": article_data.get("PublicationType"),
                "Campo": row['Campo'],
                "Pergunta": row['Pergunta'],
                "PerguntaContextualizada": contextualized_question,
                "Resposta": answer
            })

    df_answers = pd.DataFrame(all_answers)
    df_answers.to_csv("answers_by_article.csv", index=False)

    print("\n Answers saved in answers_by_article.csv.")
    print("Finished at :", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    print("Goodbye..")


   