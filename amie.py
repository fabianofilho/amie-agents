from fastapi import FastAPI
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import streamlit as st
import requests
import time

# Configuração do Streamlit
st.set_page_config(page_title="Medical AI Agents", layout="wide")

# Inicializa a API FastAPI
app = FastAPI()

# Inicializa o modelo Ollama com configurações específicas
ollama_model = "gemma:1b"  # Usando o modelo Gemma 1B
llm = OllamaLLM(
    model=ollama_model,
    base_url="http://localhost:11434",  # URL padrão do Ollama
    temperature=0.7,
    num_ctx=2048,
    num_thread=4  # Aumentando o número de threads para melhor performance
)

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
def dialogue_agent(data: PatientInput):
    prompt = PromptTemplate(
        input_variables=["symptoms", "history"],
        template="""Você é um médico assistente. O paciente apresenta os seguintes sintomas: {symptoms}
        Histórico médico: {history}
        
        Por favor, faça perguntas relevantes para coletar mais informações sobre o caso do paciente.
        Mantenha um tom profissional e empático."""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    response = chain.run({"symptoms": data.symptoms, "history": data.history})
    return {"agent": "dialogue", "response": response}

# Agente 2: Raciocínio Clínico
@app.post("/clinical_reasoning")
def clinical_reasoning_agent(data: PatientInput):
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
    return {"agent": "clinical_reasoning", "response": response}

# Agente 3: Prescrição Segura
@app.post("/medication_safety")
def medication_safety_agent(data: PatientInput):
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
    return {"agent": "medication_safety", "response": response}

# Rota para verificar se o servidor está rodando
@app.get("/")
def read_root():
    return {"message": "Medical AI Agents API is running!"}

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
                
                # Executa os agentes
                dialogue_response = requests.post("http://127.0.0.1:8000/dialogue", json=input_data).json()
                reasoning_response = requests.post("http://127.0.0.1:8000/clinical_reasoning", json=input_data).json()
                safety_response = requests.post("http://127.0.0.1:8000/medication_safety", json=input_data).json()
                
                # Exibe resultados
                st.subheader("📝 Perguntas Adicionais")
                st.write(dialogue_response["response"])
                
                st.subheader("🔍 Análise Clínica")
                st.write(reasoning_response["response"])
                
                st.subheader("💊 Segurança Medicamentosa")
                st.write(safety_response["response"])
                
            except requests.exceptions.ConnectionError:
                st.error("Erro de conexão com o servidor. Certifique-se de que o servidor FastAPI está rodando!")
            except Exception as e:
                st.error(f"Ocorreu um erro: {str(e)}")

if __name__ == "__main__":
    main()