import streamlit as st
from g4f import Client

# Função principal da aplicação Streamlit
def main():
    # Configurações da página
    st.set_page_config(page_title="Chat GPT Simples", layout="centered")
    st.title("💬 Chat GPT Simples")
    st.write("Converse com o modelo `gpt-4o-mini` usando G4F.")

    # Inicializa o cliente G4F na sessão se ainda não existir
    if "cliente" not in st.session_state:
        st.session_state.cliente = Client()

    # Inicializa o histórico de mensagens na sessão se ainda não existir
    # Este histórico manterá todas as mensagens da conversa
    if "mensagens" not in st.session_state:
        st.session_state.mensagens = []

    # Exibe todas as mensagens no histórico da sessão
    # Cada mensagem é exibida no formato de balão de chat
    for mensagem in st.session_state.mensagens:
        with st.chat_message(mensagem["role"]):
            st.markdown(mensagem["content"])

    # Campo de entrada para a nova pergunta do usuário
    # st.chat_input é ideal para interfaces de chat, pois lida com o envio
    if pergunta := st.chat_input("Digite sua pergunta:"):
        # Adiciona a pergunta do usuário ao histórico de mensagens
        st.session_state.mensagens.append({"role": "user", "content": pergunta})

        # Exibe a pergunta do usuário imediatamente
        with st.chat_message("user"):
            st.markdown(pergunta)

        try:
            # Adiciona um spinner enquanto aguarda a resposta do modelo
            with st.spinner("Pensando..."):
                # Faz a chamada à API do modelo, passando todo o histórico de mensagens
                # O modelo "gpt-4o-mini" é usado conforme mencionado na descrição
                resposta = st.session_state.cliente.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.mensagens # Passa o histórico completo
                )
                conteudo = resposta.choices[0].message.content
                # Adiciona a resposta do assistente ao histórico de mensagens
                st.session_state.mensagens.append({"role": "assistant", "content": conteudo})

            # Exibe a resposta do assistente
            with st.chat_message("assistant"):
                st.markdown(conteudo)

        except Exception as e:
            # Em caso de erro, adiciona uma mensagem de erro ao histórico e a exibe
            erro_mensagem = f"Ocorreu um erro: {e}"
            st.session_state.mensagens.append({"role": "assistant", "content": erro_mensagem})
            with st.chat_message("assistant"):
                st.markdown(erro_mensagem)

# Garante que a função principal seja executada quando o script for iniciado
if __name__ == "__main__":
    main()