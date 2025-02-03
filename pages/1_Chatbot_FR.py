!pip install chromadb
import streamlit as st
import pandas as pd
import os
from pathlib import Path
import base64

# LangChain & Hugging Face
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.chains import LLMChain

import pysqlite3
import sys
sys.modules["sqlite3"] = pysqlite3

#####################
# 1. HELPER FUNCTIONS
#####################

def get_base64_of_bin_file(bin_file_path: str) -> str:
    file_bytes = Path(bin_file_path).read_bytes()
    return base64.b64encode(file_bytes).decode()

def find_parent_fr(data, r, col):
    """
    Trouve la question parente pour une ligne et colonne donnée dans le DataFrame (version FR).
    """
    i = r - 1
    parent = None
    while i >= 0 and pd.isna(parent):
        parent = data.iloc[i, col]
        i -= 1
    return parent

def create_contextual_fr(df, category, strat_id=0):
    """
    Crée un DataFrame avec questions-réponses contextuelles (version FR).
    """
    rows = []
    columns_qna = list(df.columns)

    for r, row in df.iterrows():
        for level, col in enumerate(df.columns):
            question = row[col]
            if pd.isna(question):
                continue

            # Si la question est un "leaf node"
            if level == 4 or pd.isna(row[columns_qna[level + 1]]):
                # Gérer des sous-questions multiples
                if "\n*Si" in question or "\n *" in question or "\n*" in question:
                    questions = question.replace("\n*Si", "\n*").replace("\n *", "\n*").split("\n*")
                    for subquestion in questions:
                        if len(subquestion.strip()) == 0:
                            continue

                        context = []
                        for i in range(level - 1, -1, -1):
                            parent = df.iloc[r, i]
                            if pd.isna(parent):
                                parent = find_parent_fr(df, r, i)
                            if pd.notna(parent):
                                context = [parent] + context

                        rows.append({
                            "id": strat_id + len(rows) + 1,
                            "question": " > ".join(context),
                            "answer": subquestion.strip(),
                            "category": category,
                        })
                else:
                    context = []
                    for i in range(level - 1, -1, -1):
                        parent = df.iloc[r, i]
                        if pd.isna(parent):
                            parent = find_parent_fr(df, r, i)
                        if pd.notna(parent):
                            context = [parent] + context

                    rows.append({
                        "id": strat_id + len(rows) + 1,
                        "question": " > ".join(context),
                        "answer": question.strip(),
                        "category": category,
                    })

    return pd.DataFrame(rows)

def load_excel_and_create_vectorstore_fr(excel_path: str, persist_dir: str = "./chroma_db_fr"):
    """
    Charge les données depuis plusieurs feuilles Excel (version FR),
    construit & stocke un Chroma VectorStore.
    """
    # 1. Charger les feuilles Excel 
    qna_tree_fr0 = pd.read_excel(excel_path, sheet_name="Prépayé (FR)", skiprows=1).iloc[:, :5]
    qna_tree_fr1 = pd.read_excel(excel_path, sheet_name="Postpayé (FR)", skiprows=1).iloc[:, :5]
    qna_tree_fr2 = pd.read_excel(excel_path, sheet_name="Wifi (FR)",      skiprows=1).iloc[:, :5]

    # 2. Construire le contexte
    context_fr0 = create_contextual_fr(qna_tree_fr0, "Prépayé", strat_id = 0)
    context_fr1 = create_contextual_fr(qna_tree_fr1, "Postpayé", strat_id = len(context_fr0))
    context_fr2 = create_contextual_fr(qna_tree_fr2, "Wifi",     strat_id = len(context_fr0) + len(context_fr1))

    # 3. Concaténer les DataFrame
    context_fr = pd.concat([context_fr0, context_fr1, context_fr2], axis=0)

    # 4. Créer une colonne "context"
    context_fr["context"] = context_fr.apply(
        lambda row: f"{row['question']} > {row['answer']}",
        axis=1
    )

    # 5. Convertir chaque ligne en Document
    documents_fr = [
        Document(
            page_content=row["context"],
            metadata={"id": row["id"], "category": row["category"]}
        )
        for _, row in context_fr.iterrows()
    ]

    # 6. Créer & persister le vecteur
    embedding_model_fr = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_fr = Chroma.from_documents(documents_fr, embedding_model_fr, persist_directory=persist_dir)
    vectorstore_fr.persist()

    return vectorstore_fr

def load_existing_vectorstore_fr(persist_dir: str = "./chroma_db_fr"):
    """
    Charge un VectorStore Chroma déjà stocké (version FR).
    """
    embedding_model_fr = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_fr = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model_fr
    )
    return vectorstore_fr

def retrieve_context_fr(retriever_fr, query, top_k=5):
    """
    Récupère les top_k résultats pour la question (version FR).
    """
    results_fr = retriever_fr.get_relevant_documents(query)
    context_fr_list = []
    for _, result in enumerate(results_fr[:top_k], start=1):
        context_fr_list.append(result.page_content)
    return context_fr_list


#########################
# 2. PROMPT & LLM FR   #
#########################

prompt_template_fr = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        """[SYSTEM]
Vous êtes un assistant client professionnel, expérimenté et bienveillant pour l'opérateur téléphonique INWI. 
Vous excellez dans la gestion des clients, en répondant à leurs problèmes et questions.
Fournir un service client et des conseils en se basant sur les contextes fournis :
- Répondre aux salutations de manière courtoise et amicale, par exemple : "Je suis l'assistant IA d'INWI'. Comment puis-je vous aider aujourd'hui ?" 
- Identifier le besoin du client et demander des clarifications si nécessaire, tout en s'appuyant uniquement sur le contexte.
- Si la question n'est pas liée au contexte d'INWI, veuillez informer poliment que vous ne pouvez pas répondre à des questions hors contexte INWI.
- Si la réponse ne figure pas dans le contexte, vous pouvez dire "Je n'ai pas assez d'information" et proposer d'appeler le service client au 120.
- Structurer les réponses de manière concise et efficace. Et n'inventez pas d'infos non présentes dans le contexte.
- Informer le client qu’il peut vous recontacter pour toute assistance supplémentaire.
- Ne parlez pas des concurrents qui offrent la meme service d'INWI.
- Si la réponse n'exist pas dans le context explicitement ne repons pas.
- Ne jamais insulter ou répondre à une insulte.
- Ne demandez pas d’informations personnelles ou d’identification du client.
- Orientez vers le catalogue sur le site web INWI si la question concerne une offre du catalogue.
- Donnez des solutions standard pour les problèmes techniques avec des options.
- Avant de générer votre réponse, éliminez toutes les structures comme '[Action] [texte]' et gardez uniquement les informations utiles.
- Ne jamais parler des sujets suivants : [
    "politique", "élections", "partis", "gouvernement", "lois", "réformes",
    "religion", "croyances", "pratiques religieuses", "théologie",
    "moralité", "débat", "philosophie", "éthique", "discrimination",
    "concurrence", "Maroc Telecom", "IAM", "Orange", "comparaison",
    "sécurité", "fraude", "santé", "médicaments", "traitement", "diagnostic", "maladie",
    "finance", "investissement", "bourse", "crypto", "banque", "assurance",
    "violence", "haine", "contenu explicite", "sexe", "adultes",
    "illégal", "faux documents", "streaming illégal"
]
INWI est un opérateur de télécommunications marocain offrant des services mobiles, Internet et solutions de télécommunications
pour les particuliers et les entreprises. Il se distingue par son engagement à fournir des services de qualité, innovants et 
accessibles, tout en contribuant au développement numérique du pays.
Les clients sont notre priorité, et notre but est de résoudre leurs problèmes. 
Votre rôle est de fournir un service client professionnel et efficace sans inventer d'informations.

[CONTEXTE]
{context}

[QUESTION DU CLIENT]
{query}

[RÉPONSE]"""
    )
)

# Configuration du LLM HuggingFace (FR)

llm_fr = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API"],
    model_kwargs={
        "temperature": 0.5,
        "max_length": 500
    }
)

# Chaîne FR
llm_chain_fr = LLMChain(llm=llm_fr, prompt=prompt_template_fr)


#########################
# 3. STREAMLIT MAIN APP #
#########################

def main():
    st.subheader("INWI IA Chatbot - Français")

     # Read local image and convert to Base64
    img_base64 = get_base64_of_bin_file("./img/logo inwi celeverlytics.png")
    css_logo = f"""
    <style>
    [data-testid="stSidebarNav"]::before {{
        content: "";
        display: block;
        margin: 0 auto 20px auto;
        width: 80%;
        height: 100px;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """

    st.markdown(css_logo, unsafe_allow_html=True)

    # Charger ou créer le retriever
    if "retriever_fr" not in st.session_state:
        st.session_state["retriever_fr"] = None

    st.sidebar.header("Vector Store Options (FR)")
    
    if st.sidebar.button("Créer la Vector Store (FR)"):
        with st.spinner("Extraction et création de la vector store FR..."):
            excel_path = "Chatbot myinwi.xlsx"  
            persist_directory_fr = "./chroma_db_fr"
            vectorstore_fr = load_excel_and_create_vectorstore_fr(
                excel_path=excel_path,
                persist_dir=persist_directory_fr
            )
            st.session_state["retriever_fr"] = vectorstore_fr.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.5}
            )
        st.success("Vector store FR créée et chargée avec succès !")

    if st.sidebar.button("Charger la Vector Store existante (FR)"):
        with st.spinner("Chargement de la vector store FR existante..."):
            persist_directory_fr = "./chroma_db_fr"
            vectorstore_fr = load_existing_vectorstore_fr(persist_directory_fr)
            st.session_state["retriever_fr"] = vectorstore_fr.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.5}
            )
        st.success("Vector store FR chargée avec succès !")

    st.write("""Je suis là pour répondre à toutes vos questions concernant nos 
            services, nos offres mobiles et Internet, ainsi que nos solutions adaptées à vos besoins (FR).""")

    # Zone de texte
    user_query_fr = st.chat_input("Posez votre question ici (FR)...")

    if user_query_fr:
        if not st.session_state["retriever_fr"]:
            st.warning("Veuillez d'abord créer ou charger la Vector Store (FR).")
            return
        
        # Récupération du contexte
        context_fr_list = retrieve_context_fr(st.session_state["retriever_fr"], user_query_fr, top_k=5)

        if context_fr_list:
            with st.spinner("Génération de la réponse..."):
                response_fr = llm_chain_fr.run({"context": "\n".join(context_fr_list), "query": user_query_fr})
                # Séparer si jamais le prompt contient [RÉPONSE], sinon on affiche tout
                response_fr = response_fr.split("[RÉPONSE]")[-1]
            st.write("**Question :**")
            st.write(user_query_fr)
            st.write("**Réponse :**")
            st.write(response_fr)
        else:
            st.write("Aucun contexte trouvé pour cette question. Essayez autre chose.")


if __name__ == "__main__":
    main()


