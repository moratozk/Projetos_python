# Importação das bibliotecas necessárias
from dotenv import load_dotenv
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.document_loaders import PDFMinerLoader
import os
import vertexai as vt
from tkinter.filedialog import askopenfilename
import streamlit as st
import nest_asyncio # Necessário para evitar problemas com o loop de eventos do Streamlit
import tempfile

nest_asyncio.apply() # Permite que o Streamlit funcione corretamente com asyncio

def config_IA(temp_pdf_path):
    #Variaveis globais
    global chain  # Declara a variável global para a cadeia de perguntas e respostas
    global vectorstore  # Declara a variável global para o vetor de similaridade
    global chat  # Declara a variável global para o modelo de chat
    global prompt  # Declara a variável global para o prompt

    #Inicicar projeto no modulo Vertex AI no google cloud
    vt.init(project="chatbotnb")  # Inicializa o projeto no Google Cloud

    #Configuração da chave de API do Google Cloud
    CHAVEAPI = "AIzaSyCqEOEuTxt7A8ZgfRZ3hexdB0LztScFPjY"  # Chave da API do Google Cloud
    os.environ["GOOGLE_API_KEY"] = CHAVEAPI  # Configura a chave da API

    #Autenticação da chave API no Google Cloud
    current_dir = os.path.dirname(os.path.abspath(__file__))
    credenciais = os.path.join(current_dir, "chatbotnb-00b2af081539.json")  # Caminho para o arquivo de credenciais
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credenciais  # Configura as credenciais do Google Cloud

    # Cria embeddings usando o modelo do Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Carrega as variáveis de ambiente do arquivo .env
    load_dotenv()

    CAMINHO = temp_pdf_path  # Variável para armazenar o caminho do arquivo PDF
    # CAMINHO = r"C:\Users\Professor\Documents\IA_ferias\plano.pdf"  # Variável para armazenar o caminho do arquivo PDF

    loader = PDFMinerLoader(file_path=CAMINHO)  # Carrega o documento PDF
    documentos = loader.load()  # Carrega o conteúdo do PDF

    vectorstore = FAISS.from_documents(documentos, embeddings)  # Cria o vetor de similaridade

    chat = init_chat_model(
        model="gemini-2.0-flash",
        temperature=0.2,
        max_output_tokens=1000,
        top_p=0.8,
        top_k=40,
        model_kwargs={
            "api_key": CHAVEAPI,
            "project_id": "chatbotnb",
            }
    )

    prompt = PromptTemplate(
        input_variables=["pergunta", "resposta_similar"],
        template="Vocé é um assistente de elaboração de planejamento de ensino da empresa Senai," \
        "você auxilia o professor a elaborar um planejamento de ensino de forma clara e objetiva." \
        "Você deve responder a pergunta do usuário com base no conteúdo do documento PDF carregado."
        "Pergunta: {pergunta}\n\n" \
        "Resposta similar: {resposta_similar}\n\n" \
        "Resposta:"
    )
    chain = LLMChain(
        llm=chat,
        prompt=prompt,
    )

# config_IA()

def receber_informacao(pergunta):
    resposta_similar = vectorstore.similarity_search(pergunta, k=1)  # Busca a resposta mais similar
    return [vectorstore.page_content for vectorstore in resposta_similar]  # Retorna o conteúdo da resposta similar

def responder_pergunta(pergunta):
    resposta_similar = receber_informacao(pergunta)  # Recebe a informação similar
    resposta = chain.run(pergunta=pergunta, resposta_similar=resposta_similar[0])  # Executa a cadeia de perguntas e respostas
    return resposta  # Retorna a resposta gerada pela IA

#
    # Abre um diálogo para selecionar o arquivo PDF
    global pdf  # Declara a variável global para o caminho do PDF
    pdf = st.file_uploader("Selecione um arquivo PDF", type=["pdf"])  # Permite ao usuário selecionar um arquivo PDF
    # Verifica se um arquivo PDF foi selecionado, se não, encerra o programa
    if not pdf:
        st.error("Nenhum arquivo PDF selecionado. Encerrando o programa.")
        raise Exception("Nenhum arquivo PDF selecionado.")

def streamlit_app():
    st.title("Professor Mestre Gilberto seu assistente pessoal de planejamento de ensino")
    st.write("Digite a sua pergunta ou escreva sair para encerrar o chat")

    #selecionar_pdf()  # Chama a função para selecionar o PDF
    entrada_usuario = st.text_input("Pergunta:")
   
    st.sidebar.title("Carregar PDF")

    with st.sidebar:
        st.markdown("### 📄 Carregue o seu PDF de planejamento:")
        pdf = st.file_uploader("Selecionar PDF", type=["pdf"])
        if pdf:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf.read())
                temp_pdf_path = tmp.name

            st.success("✅ PDF carregado com sucesso!")
            config_IA(temp_pdf_path)
        elif "pdf_path" not in st.session_state:
            st.warning("⚠️ Por favor, carregue um PDF para começar.")
            st.stop()

    if st.button("Enviar"):
        if entrada_usuario:
            resposta_bot = responder_pergunta(entrada_usuario)
            st.write(f"Gilberto: {resposta_bot}")
            st.chat_message(resposta_bot)  # Exibe a resposta do assistente

        else:
            st.warning("Por favor, digite uma pergunta antes de enviar.")

if __name__ == "__main__":
    streamlit_app()  # Inicia o aplicativo Streamlit
     # Chama a função para configurar a IA
    #main()  # Inicia o chat com o usuário