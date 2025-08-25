import os
from dotenv import load_dotenv
import streamlit as st

from decouple import config

from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits.sql.toolkit  import SQLDatabaseToolkit
from langchain_openai import ChatOpenAI
# from langchain.chat_models import ChatOpenAI



os.environ['OPENAI_API_KEY'] = config('OPENAI_API_KEY')

st.set_page_config(
    page_title='Estoque GPT',
    page_icon='📄',
)
st.header('Assistente de Estoque')

model_options = [
    'gpt-3.5-turbo',
    'gpt-4',
    'gpt-4-turbo',
    'gpt-4o-mini',
    'gpt-4o'
]

selected_model = st.sidebar.selectbox(
    label = 'Selecione o modelo LLM',
    options = model_options
)

st.sidebar.markdown('### Sobre')
st.sidebar.markdown('Este agente consulta um banco de dados de estoque utilizando um modelo GPT.')

st.write('Faça perguntas sobre o estoque de produtos, preços e reposições.')
user_question = st.text_input('O que deseja saber sobre o estoque?')

load_dotenv() # carrega as variáveis do .env

# CRIAÇÃO DO AGENTE
model = ChatOpenAI(
    model = selected_model,
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    max_retries=5,   # tenta novamente se der RateLimit ou erro de rede
    temperature=0    # opcional: mais previsível p/ queries SQL
)

# CONEXÃO COM O BANCO
db = SQLDatabase.from_uri('sqlite:///estoque.db')
toolkit = SQLDatabaseToolkit(
    db = db,
    llm = model
)
system_message = hub.pull('hwchase17/react')

# CRIANDO AGENTE
agent = create_react_agent(
    llm = model,
    tools = toolkit.get_tools(),
    prompt = system_message
)

# CRIANDO AGENTE EXECUTOR
agent_executor = AgentExecutor(
    agent = agent,
    tools = toolkit.get_tools(),
    verbose = True
)

# PROMPT
prompt = '''
    Use as ferramentas necessárias para responder perguntas relacionadas ao estoque de produtos. Você fornecerá insights sobre produtos, preços, reposição de estoque e relatórios conforme solicitado pelo usuário. A resposta final deve ser uma formatação amigável de visualização para o usuário. Sempre responda em português brasileiro.
    Pergunta: {q}
'''
prompt_template = PromptTemplate.from_template(prompt)

# CONDIÇÃO DO BOTÃO
if st.button('Consultar'):
    if user_question:
        with st.spinner('Consultando o Banco de Dados...'):
            formatted_prompt = prompt_template.format(q=user_question)
            output = agent_executor.invoke({'input': formatted_prompt})
            resposta = output.get("output", "Não foi possível gerar uma resposta.")
            st.markdown(resposta)
    else:
        st.warning('Por favor, insira uma pergunta.')