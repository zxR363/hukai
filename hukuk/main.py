import sys

# Çıktı karakter hatası olmasın diye
sys.stdout.reconfigure(encoding='utf-8')

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- %100 LOKAL KÜTÜPHANELER ---
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama  # <-- Yeni oyuncumuz bu


def create_local_cv_bot():
    print("--- 1. PDF Yükleniyor... ---")
    try:
        loader = PyPDFLoader("Yusuf_Ustuntepe_CV_tr.pdf")
        docs = loader.load()
    except Exception as e:
        print(f"HATA: PDF bulunamadı. Detay: {e}")
        return None

    print("--- 2. Metin Parçalanıyor... ---")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    print("--- 3. Vektör Veritabanı (Local CPU)... ---")
    # Embedding: Metni sayıya çeviren kısım (HuggingFace - Local)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    print("--- 4. LLM (Ollama) Bağlanıyor... ---")
    # LLM: Cevabı veren kısım (Ollama - Local)
    # 'llama3.2' modelini az önce terminalden indirdik.
    llm = ChatOllama(model="llama3.2", temperature=0)

    system_prompt = (
        "Sen teknik bir işe alım asistanısın. "
        "Aşağıdaki CV içeriğine dayanarak soruları Türkçe cevapla. "
        "Bilmediğin bir şey sorulursa uydurma. "
        "\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    # Zinciri oluşturuyoruz
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    return rag_chain


if __name__ == "__main__":
    print("\n🖥️ SİSTEM BAŞLATILIYOR (Lokal Mod)...")
    bot = create_local_cv_bot()

    if bot:
        print("\n✅ OLLAMA BOT HAZIR! (İnternet bağlantısı gerekmez)")
        print("Çıkmak için 'q' yazın.\n")

        while True:
            try:
                soru = input("Soru Sor: ")
                if soru.lower() == 'q':
                    break

                print("⏳ Düşünüyor (İşlemcinize bağlı olarak biraz sürebilir)...")
                response = bot.invoke({"input": soru})
                print(f"\nCEVAP: {response['answer']}")
                print("-" * 50)
            except Exception as e:
                print(f"Hata: {e}")
                print("İPUCU: 'ollama' uygulamasının arka planda açık olduğundan emin misin?")