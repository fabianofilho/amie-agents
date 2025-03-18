from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import time
import json
import logging
import os

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuração do Streamlit
st.set_page_config(page_title="Medical AI Agents", layout="wide")

# Inicializa a API FastAPI
app = FastAPI()

# Verifica se o Ollama está rodando
def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code == 200:
            logger.info("Ollama está rodando e acessível")
            return True
        else:
            logger.error(f"Ollama retornou status code {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("Não foi possível conectar ao Ollama. Certifique-se de que ele está rodando.")
        return False

# Inicializa o modelo Ollama com configurações específicas
ollama_model = "gemma3"  # Usando o modelo Gemma 3
try:
    if not check_ollama():
        raise Exception("Ollama não está rodando ou não está acessível")

    llm = OllamaLLM(
        model=ollama_model,
        base_url="http://localhost:11434",
        temperature=0.7,
        num_ctx=32768,  # Contexto maior para o Gemma 3
        num_thread=4,
        timeout=300,  # Aumentando o timeout para 300 segundos
        num_gpu=1,  # Usando GPU se disponível
        num_batch=512,  # Aumentando o tamanho do batch
        repeat_penalty=1.1,  # Penalidade para repetições
        stop=["<end_of_turn>"],  # Token de parada
        seed=42  # Semente fixa para consistência
    )
    logger.info(f"Modelo Ollama {ollama_model} inicializado com sucesso")
except Exception as e:
    logger.error(f"Erro ao inicializar o modelo Ollama: {str(e)}")
    raise

# Modelo de entrada para os agentes
class PatientInput(BaseModel):
    patient_id: str
    symptoms: str
    history: str
    medications: str

# Função para carregar exemplo do arquivo
def load_example():
    try:
        with open('example.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            # Parse do conteúdo
            lines = content.split('\n')
            patient_data = {}
            for line in lines:
                if ':' in line:
                    key, value = line.split(':', 1)
                    patient_data[key.strip()] = value.strip()
            return patient_data
    except Exception as e:
        st.error(f"Erro ao carregar exemplo: {str(e)}")
        return None

# Agente 1: Diálogo Médico
@app.post("/dialogue")
async def dialogue_agent(data: PatientInput):
    try:
        logger.info(f"Iniciando diálogo para paciente {data.patient_id}")
        prompt = PromptTemplate(
            input_variables=["symptoms", "history"],
            template="""Você é um médico assistente. O paciente apresenta os seguintes sintomas: {symptoms}
            Histórico médico: {history}
            
            Por favor, faça perguntas relevantes para coletar mais informações sobre o caso do paciente.
            Mantenha um tom profissional e empático."""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"symptoms": data.symptoms, "history": data.history})
        logger.info("Diálogo concluído com sucesso")
        return {"agent": "dialogue", "response": response}
    except Exception as e:
        logger.error(f"Erro no agente de diálogo: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Agente 2: Raciocínio Clínico
@app.post("/clinical_reasoning")
async def clinical_reasoning_agent(data: PatientInput):
    try:
        logger.info(f"Iniciando raciocínio clínico para paciente {data.patient_id}")
        prompt = PromptTemplate(
            input_variables=["symptoms", "history"],
            template="""Baseado nos seguintes sintomas: {symptoms}
            E no histórico médico: {history}
            
            Por favor, forneça:
            1. Possíveis diagnósticos diferenciais
            2. Exames complementares recomendados
            3. Plano de tratamento inicial
            4. Recomendações ao paciente
            
            Baseie suas respostas em guidelines médicas atuais."""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"symptoms": data.symptoms, "history": data.history})
        logger.info("Raciocínio clínico concluído com sucesso")
        return {"agent": "clinical_reasoning", "response": response}
    except Exception as e:
        logger.error(f"Erro no agente de raciocínio clínico: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Agente 3: Prescrição Segura
@app.post("/medication_safety")
async def medication_safety_agent(data: PatientInput):
    try:
        logger.info(f"Iniciando análise de segurança medicamentosa para paciente {data.patient_id}")
        prompt = PromptTemplate(
            input_variables=["medications", "history"],
            template="""Analise as seguintes medicações em uso: {medications}
            Considerando o histórico médico: {history}
            
            Por favor, avalie:
            1. Possíveis interações medicamentosas
            2. Contraindicações
            3. Recomendações de ajustes
            4. Monitoramento necessário"""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run({"medications": data.medications, "history": data.history})
        logger.info("Análise de segurança medicamentosa concluída com sucesso")
        return {"agent": "medication_safety", "response": response}
    except Exception as e:
        logger.error(f"Erro no agente de segurança medicamentosa: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Rota para verificar se o servidor está rodando
@app.get("/")
async def read_root():
    return {"status": "ok", "message": "Medical AI Agents API is running!"}

# Interface Streamlit
def main():
    st.title("🤖 Assistente Médico Inteligente")
    st.markdown("""
    Este assistente utiliza três agentes especializados para auxiliar na avaliação médica:
    1. **Agente de Diálogo**: Faz perguntas relevantes para coletar mais informações
    2. **Agente de Raciocínio Clínico**: Fornece análise diagnóstica e plano de tratamento
    3. **Agente de Segurança Medicamentosa**: Avalia interações e contraindicações
    """)

    # Botão para carregar exemplo
    if st.button("📋 Carregar Exemplo"):
        example_data = load_example()
        if example_data:
            st.session_state.patient_id = example_data.get('ID do Paciente', '')
            st.session_state.symptoms = example_data.get('Sintomas', '')
            st.session_state.history = example_data.get('Histórico Médico', '')
            st.session_state.medications = example_data.get('Medicações Atuais', '')
            st.success("Exemplo carregado com sucesso!")

    # Campos de entrada
    col1, col2 = st.columns(2)
    
    with col1:
        patient_id = st.text_input("ID do Paciente", 
                                 value=st.session_state.get('patient_id', ''))
        symptoms = st.text_area("Sintomas", 
                              value=st.session_state.get('symptoms', ''),
                              height=100)
    
    with col2:
        history = st.text_area("Histórico Médico", 
                             value=st.session_state.get('history', ''),
                             height=100)
        medications = st.text_area("Medicações Atuais", 
                                 value=st.session_state.get('medications', ''),
                                 height=100)

    if st.button("Executar Análise", type="primary"):
        if not all([patient_id, symptoms, history, medications]):
            st.error("Por favor, preencha todos os campos!")
            return

        with st.spinner("Processando informações..."):
            try:
                input_data = {
                    "patient_id": patient_id,
                    "symptoms": symptoms,
                    "history": history,
                    "medications": medications
                }
                
                # Verifica se o servidor está rodando
                try:
                    health_check = requests.get("http://127.0.0.1:8000/", timeout=5)
                    if health_check.status_code != 200:
                        st.error(f"Servidor retornou status code {health_check.status_code}")
                        return
                except requests.exceptions.ConnectionError:
                    st.error("Servidor não está rodando. Por favor, inicie o servidor usando 'python run_server.py'")
                    return
                
                # Executa os agentes com timeout e verificação de status
                for endpoint in ["dialogue", "clinical_reasoning", "medication_safety"]:
                    try:
                        response = requests.post(
                            f"http://127.0.0.1:8000/{endpoint}",
                            json=input_data,
                            timeout=300  # Aumentando o timeout para 300 segundos
                        )
                        response.raise_for_status()
                        result = response.json()
                        
                        if endpoint == "dialogue":
                            st.subheader("📝 Perguntas Adicionais")
                            st.write(result.get("response", "Sem resposta disponível"))
                        elif endpoint == "clinical_reasoning":
                            st.subheader("🔍 Análise Clínica")
                            st.write(result.get("response", "Sem resposta disponível"))
                        else:
                            st.subheader("💊 Segurança Medicamentosa")
                            st.write(result.get("response", "Sem resposta disponível"))
                            
                    except requests.exceptions.Timeout:
                        st.error(f"Tempo limite excedido ao processar {endpoint}. O servidor está demorando muito para responder.")
                        logger.error(f"Timeout ao processar {endpoint}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Erro ao processar {endpoint}: {str(e)}")
                        logger.error(f"Erro detalhado para {endpoint}: {str(e)}")
                
            except Exception as e:
                st.error(f"Ocorreu um erro: {str(e)}")
                logger.error(f"Erro detalhado: {str(e)}")

if __name__ == "__main__":
    main()