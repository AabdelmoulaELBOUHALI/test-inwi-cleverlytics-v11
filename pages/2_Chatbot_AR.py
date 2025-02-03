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

####
from langchain_openai import ChatOpenAI

#####################
# 1. HELPER FUNCTIONS
#####################

def get_base64_of_bin_file(bin_file_path: str) -> str:
    file_bytes = Path(bin_file_path).read_bytes()
    return base64.b64encode(file_bytes).decode()

def find_parent_ar(data, r, col):
    """
     Trouve la question parente pour une ligne et colonne donnée dans le DataFrame (version AR).
    """
    i = r - 1
    parent = None
    while i >= 0 and pd.isna(parent):
        parent = data.iloc[i, col]
        i -= 1
    return parent

def create_contextual_ar(df, category, strat_id=0):
    """
    Crée un DataFrame avec questions-réponses contextuelles (version AR).
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
                                parent = find_parent_ar(df, r, i)
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
                            parent = find_parent_ar(df, r, i)
                        if pd.notna(parent):
                            context = [parent] + context

                    rows.append({
                        "id": strat_id + len(rows) + 1,
                        "question": " > ".join(context),
                        "answer": question.strip(),
                        "category": category,
                    })

    return pd.DataFrame(rows)

def load_excel_and_create_vectorstore_ar(excel_path: str, persist_dir: str = "./chroma_db_ar"):
    """
    Charge les données depuis plusieurs feuilles Excel (version AR),
    construit & stocke un Chroma VectorStore.
    """
    # 1. Charger les feuilles Excel 
    qna_tree_ar0 = pd.read_excel(excel_path, sheet_name="Prépayé (AR)", skiprows=1).iloc[:, :5]
    qna_tree_ar1 = pd.read_excel(excel_path, sheet_name="Postpayé (AR)", skiprows=1).iloc[:, :5]
    qna_tree_ar2 = pd.read_excel(excel_path, sheet_name="Wifi (AR)",      skiprows=1).iloc[:, :5]

    # 2. Construire le contexte
    context_ar0 = create_contextual_ar(qna_tree_ar0, "دفع مسبق", strat_id = 0)
    context_ar1 = create_contextual_ar(qna_tree_ar1, "دفع لاحق", strat_id = len(context_ar0))
    context_ar2 = create_contextual_ar(qna_tree_ar2, "واي فاي",   strat_id = len(context_ar0) + len(context_ar1))

    # 3. Concaténer les DataFrame
    context_ar = pd.concat([context_ar0, context_ar1, context_ar2], axis=0)

    # 4. Créer une colonne "context"
    context_ar["context"] = context_ar.apply(
        lambda row: f"{row['question']} > {row['answer']}",
        axis=1
    )

    # 5. Convertir chaque ligne en Document
    documents_ar = [
        Document(
            page_content=row["context"],
            metadata={"id": row["id"], "category": row["category"]}
        )
        for _, row in context_ar.iterrows()
    ]

    # 6. Créer & persister le vecteur
    embedding_model_ar = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_ar = Chroma.from_documents(documents_ar, embedding_model_ar, persist_directory=persist_dir)
    vectorstore_ar.persist()

    return vectorstore_ar

def load_existing_vectorstore_ar(persist_dir: str = "./chroma_db_ar"):
    """
    Charge un VectorStore Chroma déjà stocké (version AR).
    """
    embedding_model_ar = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore_ar = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model_ar
    )
    return vectorstore_ar

def retrieve_context_ar(retriever_ar, query, top_k=5):
    """
    Récupère les top_k résultats pour la question (version AR).
    """
    results_ar = retriever_ar.get_relevant_documents(query)
    context_ar_list = []
    for _, result in enumerate(results_ar[:top_k], start=1):
        context_ar_list.append(result.page_content)
    return context_ar_list


#########################
# 2. PROMPT & LLM (AR) #
#########################

prompt_template_ar = PromptTemplate(
    input_variables=["context", "query"],
    template=(
        """[SYSTEM]
أنت مساعد لخدمة عملاء INWI، محترف وخبير ومتعاون. تتقن التعامل مع استفسارات ومشاكل العملاء.
استند فقط إلى المعلومات المتوفرة في السياقات التالية دون اختراع معلومات غير موجودة:
- استخدم تحية مهذبة وودّية، على سبيل المثال: "مرحباً، أنا المساعد الذكي من إنوي. كيف يمكنني خدمتك اليوم؟"
- تعرّف على احتياج العميل واطلب التوضيح إذا لزم الأمر بالاعتماد على المعلومات المتوفرة فقط.
- إن لم يكن السؤال ضمن سياق إنوي، أخبر العميل بلطف أنك غير قادر على الإجابة خارج سياق إنوي.
- إذا لم تجد إجابة واضحة في السياق، يمكنك إبلاغ العميل بعدم توفر المعلومات واقتراح الاتصال بخدمة العملاء على الرقم 120.
- احرص على أن تكون ردودك موجزة وفعالة. وتجنّب اختلاق أي تفاصيل غير موجودة في السياق.
- أخبر العميل بأنه يمكنه التواصل معك مجدداً لمزيد من المساعدة.
- لا تتحدث عن المنافسين الذين يقدمون نفس خدمات إنوي.
- امتنع تماماً عن أي إهانة أو رد على إهانة.
- لا تطلب أي معلومات شخصية أو هوية العميل.
- وجّه العميل إلى كتالوج موقع إنوي إذا كان سؤاله يتعلق بعروض من الكتالوج.
- قدّم حلولاً قياسية للمشكلات التقنية مع عرض الخيارات المتاحة.
- قبل إرسال الجواب، تجنب أي تنسيق مثل "[Action] [نص]" واحتفظ فقط بالمعلومات المفيدة.
- لا تتحدث عن المواضيع التالية إطلاقاً: [
    "السياسة", "الانتخابات", "الأحزاب", "الحكومة", "القوانين", "الإصلاحات",
    "الدين", "العقائد", "الممارسات الدينية", "علم اللاهوت",
    "الأخلاق", "الجدل", "الفلسفة", "المعايير", "التمييز",
    "المنافسة", "مقارنة إنوي مع شركات أخرى",
    "الأمن", "الاحتيال", "الصحة", "الأدوية", "التشخيص الطبي",
    "التمويل", "الاستثمار", "البورصة", "العملات الرقمية", "البنوك", "التأمين",
    "العنف", "الكراهية", "المحتوى الفاضح", "الجنس",
    "المخالفات القانونية", "الوثائق المزورة", "البث غير الشرعي"
]
إنوي (INWI) هي شركة اتصالات مغربية تقدم خدمات الهاتف المحمول والإنترنت وحلول الاتصالات للأفراد والشركات.
تتميز بالتزامها بتوفير خدمات عالية الجودة ومبتكرة، والمساهمة في التطور الرقمي في المغرب.
العملاء هم أولويتنا، وهدفنا مساعدتهم وحل مشاكلهم.
دورك هو تقديم خدمة عملاء احترافية وفعالة بدون اختراع معلومات من خارج السياق.

[السياق]
{context}

[سؤال العميل]
{query}

[الإجابة]"""
    )
)

# Configuration du LLM HuggingFace (AR)
#os.environ["HUGGINGFACEHUB_API"]
from langchain_openai import ChatOpenAI

llm_ar = ChatOpenAI(
    model="Atlas-Chat-9B",
    base_url="https://api.friendli.ai/dedicated/v1",
    api_key=os.environ["FRIENDLI_TOKEN"],
)

# llm_ar = HuggingFaceHub(
#     repo_id="MBZUAI-Paris/Atlas-Chat-2B", #"MBZUAI-Paris/Atlas-Chat-9B",
#     huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API"],
#     model_kwargs={
#         "temperature": 0.5,
#         "max_length": 500,
#         "timeout": 600
#     }
# )

# Chaîne AR
llm_chain_ar = LLMChain(llm=llm_ar, prompt=prompt_template_ar)


#########################
# 3. STREAMLIT MAIN APP #
#########################

def main():
    st.subheader("INWI IA Chatbot - Arabe")

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

    if "retriever_ar" not in st.session_state:
        st.session_state["retriever_ar"] = None

    st.sidebar.subheader("Vector Store Options (AR)")

    if st.sidebar.button("Créer la Vector Store (AR)"):
        with st.spinner("Extraction et création de la vector store AR..."):
            excel_path = "Chatbot myinwi.xlsx"  
            persist_directory_ar = "./chroma_db_ar"
            vectorstore_ar = load_excel_and_create_vectorstore_ar(
                excel_path=excel_path,
                persist_dir=persist_directory_ar
            )
            st.session_state["retriever_ar"] = vectorstore_ar.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.5}
            )
        st.success("Vector store FR créée et chargée avec succès !")

    if st.sidebar.button("Charger la Vector Store existante (AR)"):
        with st.spinner("Chargement de la vector store FR existante..."):
            persist_directory_ar = "./chroma_db_ar"
            vectorstore_ar = load_existing_vectorstore_ar(persist_directory_ar)
            st.session_state["retriever_ar"] = vectorstore_ar.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "lambda_mult": 0.5}
            )
        st.success("Vector store AR chargée avec succès !")

    st.write("""مرحباً! أنا هنا للإجابة على جميع أسئلتك المتعلقة بخدمات إنوي 
            وعروض الهاتف المحمول والإنترنت، وأي حلول أخرى قد تناسب احتياجاتك (AR).""")

    user_query_ar = st.chat_input("Posez votre question ici (AR)...")

    if user_query_ar:
        if not st.session_state["retriever_ar"]:
            st.warning("Veuillez d'abord créer ou charger la Vector Store (AR).")
            return
        
        # Récupération du contexte
        context_ar_list = retrieve_context_ar(st.session_state["retriever_ar"], user_query_ar, top_k=3)

        if context_ar_list:
            with st.spinner("Génération de la réponse..."):
                response_ar = llm_chain_ar.run({"context": "\n".join(context_ar_list), "query": user_query_ar + "?"})
                response_ar = response_ar.split("[الإجابة]")[-1]
            st.write("**سؤال العميل:**")
            st.write(user_query_ar)
            st.write("**الإجابة:**")
            st.write(response_ar)
        else:
            st.write("Aucun contexte trouvé pour cette question. Essayez autre chose.")

if __name__ == "__main__":
    main()
