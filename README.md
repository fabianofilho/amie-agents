# AMIE Agents: Assistentes Médicos Inteligentes com MedGemma

Este repositório contém uma implementação de agentes médicos conversacionais inspirados na arquitetura do **AMIE (Articulate Medical Intelligence Explorer)** do Google DeepMind, utilizando os modelos mais recentes da família **MedGemma** (Google Health AI Developer Foundations).

## 🚀 O que há de novo (Atualização 2025/2026)

A pesquisa em IA médica avançou significativamente com as publicações recentes do Google DeepMind sobre o AMIE e o lançamento do MedGemma. Este repositório foi atualizado para refletir esses avanços:

### 1. Arquitetura Multi-Agente (Inspirada no AMIE Longitudinal)
O AMIE evoluiu de um modelo de diagnóstico pontual para um sistema de gestão longitudinal de doenças. Nossa implementação reflete essa abordagem com múltiplos agentes especializados:
- **Agente de Diálogo (Dialogue Agent)**: Focado na interação com o paciente, coleta de histórico e empatia.
- **Agente de Raciocínio Clínico (Mx Agent)**: Focado em diagnósticos diferenciais e planos de manejo baseados em guidelines.
- **Agente de Segurança Medicamentosa**: Focado em interações e contraindicações.

### 2. Integração com MedGemma (Gemma 3)
Substituímos os modelos genéricos pela família **MedGemma**, a coleção de modelos abertos do Google otimizados especificamente para compreensão de textos e imagens médicas.
- Suporte para **MedGemma 1.5 (4B Multimodal)** para análise de exames (Raio-X, ressonância, etc.).
- Suporte para **MedGemma 27B** para raciocínio clínico complexo e compreensão de prontuários (EHR).

### 3. Physician-Centered Oversight (Supervisão Médica)
Alinhado com os estudos clínicos de mundo real do AMIE (2025/2026), a arquitetura foi desenhada para atuar como um "Clinician Cockpit", onde a IA realiza a triagem e gera notas SOAP, mas a decisão final e validação permanecem com o médico humano.

---

## 🛠️ Tecnologias Utilizadas

- **Modelos Base**: MedGemma 4B/27B (via Ollama ou Hugging Face Transformers)
- **Framework de Agentes**: LangChain / CrewAI
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **Inferência Local**: Ollama (para privacidade e segurança de dados de saúde)

## 📦 Instalação e Configuração

### Pré-requisitos
- Python 3.9+
- [Ollama](https://ollama.ai/) instalado localmente (para inferência segura e privada)

### Passo a passo

1. Clone o repositório:
```bash
git clone https://github.com/fabianofilho/amie-agents.git
cd amie-agents
```

2. Crie um ambiente virtual e instale as dependências:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou venv\Scripts\activate no Windows
pip install -r requirements.txt
```

3. Baixe o modelo MedGemma no Ollama:
```bash
# Para o modelo de 4B (mais leve, ideal para testes locais)
ollama run hf.co/mlx-community/medgemma-4b-it-6bit

# Ou para o modelo de 27B (requer mais VRAM, melhor raciocínio clínico)
ollama run hf.co/mlx-community/medgemma-27b-it-bf16
```
*Nota: Você também pode usar a integração direta com Hugging Face Transformers se preferir rodar via PyTorch/Accelerate.*

## 🏃‍♂️ Como Executar

1. Inicie o servidor FastAPI (Backend dos Agentes):
```bash
python run_server.py
```

2. Em outro terminal, inicie a interface Streamlit:
```bash
streamlit run amie.py
```

3. Acesse `http://localhost:8501` no seu navegador.

## 📚 Referências e Leituras Recomendadas

1. **Towards conversational diagnostic artificial intelligence** (Nature, 2025) - O paper original do AMIE.
2. **AMIE gains vision: A research AI agent for multimodal diagnostic dialogue** (Google Research Blog, 2025) - Introdução das capacidades multimodais.
3. **From diagnosis to treatment: Advancing AMIE for longitudinal disease management** (Google Research Blog, 2025) - A arquitetura de dois agentes (Dialogue e Mx Agent).
4. **Towards physician-centered oversight of conversational diagnostic AI** (arXiv, 2025) - O framework de supervisão médica e geração de notas SOAP.
5. **MedGemma Model Card** (Google Health AI Developer Foundations) - Documentação oficial dos modelos MedGemma.

## ⚠️ Aviso Legal (Disclaimer)

Este projeto é estritamente para fins de **pesquisa e desenvolvimento educacional**. Os modelos de IA (incluindo MedGemma) não são dispositivos médicos regulamentados e **não devem ser usados para diagnóstico, tratamento ou aconselhamento médico real** sem a supervisão de um profissional de saúde qualificado. Sempre valide as saídas geradas pela IA.
