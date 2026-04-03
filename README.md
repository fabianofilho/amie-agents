<div align="center">

# AMIE Agents

**Assistentes Médicos Inteligentes com Gemma 4**
Sistema multi-agente de IA conversacional médica inspirado na arquitetura AMIE do Google DeepMind, com supervisão médica integrada.

<br/>

<a href="https://huggingface.co/spaces/fabianonbfilho/amie-agents">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Abrir%20Demo-Hugging%20Face%20Spaces-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" alt="Abrir no Hugging Face Spaces"/>
</a>

<br/><br/>

<a href="https://ai.google.dev/gemma/docs/core/model_card_4">
  <img src="https://img.shields.io/badge/Gemma%204-E4B%20%7C%2026B%20MoE%20%7C%2031B-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="Gemma 4"/>
</a>
&nbsp;
<a href="https://deepmind.google/technologies/gemini/">
  <img src="https://img.shields.io/badge/Google-DeepMind-EA4335?style=for-the-badge&logo=google-deepmind&logoColor=white" alt="Google DeepMind"/>
</a>
&nbsp;
<a href="https://ollama.ai/">
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge&logo=ollama&logoColor=white" alt="Ollama"/>
</a>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.9%2B-00D098?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-backend-00D098?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-frontend-00D098?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/LangChain-agents-00D098?style=flat-square&logo=langchain&logoColor=white)](https://www.langchain.com/)
[![Local & Privado](https://img.shields.io/badge/dados-100%25%20local-00D098?style=flat-square&logo=shield&logoColor=white)](#)
[![Português](https://img.shields.io/badge/idioma-Portugu%C3%AAs-00D098?style=flat-square)](#)

</div>

---

## O que há de novo (2025/2026)

A pesquisa em IA médica avançou significativamente com as publicações recentes do Google DeepMind sobre o AMIE e o lançamento do Gemma 4. Este repositório reflete esses avanços.

### Gemma 4 (Abril 2026)

- **Context window**: 128K–256K tokens (antes 32K)
- **Multimodal nativo**: texto, imagem e áudio
- **140+ idiomas** com suporte aprimorado para português
- **Apache 2.0** — licença totalmente aberta
- **Function calling** e modos de raciocínio (thinking)
- **MoE (26B-A4B)**: 26B parâmetros totais, ativa apenas 3.8B na inferência

### Arquitetura Multi-Agente

Inspirada no AMIE Longitudinal, a implementação utiliza três agentes especializados com supervisão mdica integrada ("Clinician Cockpit"):

| Agente | Função | Descrição |
|--------|--------|-----------|
| **Dialogue Agent** | Interação com paciente | Coleta de histórico, empatia e anamnese estruturada |
| **Mx Agent** | Raciocínio clínico | Diagnósticos diferenciais e planos de manejo baseados em guidelines |
| **Safety Agent** | Segurança medicamentosa | Análise de interações e contraindicações |

### Integração com Gemma 4

| Modelo | Uso | Descrição |
|--------|-----|-----------|
| **Gemma 4 E4B** | Testes locais (Ollama) | 4B efetivos, edge/mobile, contexto 128K, Apache 2.0 |
| **Gemma 4 31B Dense** | Google AI (cloud) | 31B denso, máxima qualidade, contexto 256K |
| **Gemma 4 26B MoE** | Eficiência + qualidade | 26B total, ativa 3.8B durante inferência |
| **MedGemma 4B** *(fallback)* | Fine-tune médico | Baseado no Gemma 3, análise de exames (Raio-X, ressonância) |

### Physician-Centered Oversight

A IA realiza triagem e gera notas SOAP, mas a decisão final e validação permanecem com o mdico humano — alinhado com os estudos clínicos de mundo real do AMIE (2025/2026).

---

## Tecnologias

| Componente | Tecnologia |
|------------|-----------|
| Modelos Base | Gemma 4 E4B/26B-MoE/31B (via Ollama ou Google AI) |
| Framework de Agentes | LangChain / CrewAI |
| Backend | FastAPI |
| Frontend | Streamlit |
| Inferência Local | Ollama (privacidade e segurança de dados) |

---

## Instalação e Configuração

### Pré-requisitos

- Python 3.9+
- [Ollama](https://ollama.ai/) instalado localmente

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

3. Baixe o modelo Gemma 4 no Ollama:
```bash
# Gemma 4 E4B (leve, ideal para testes locais — 128K contexto)
ollama run hf.co/mlx-community/gemma-4-E4B-it-6bit

# Gemma 4 MoE 26B (ativa 3.8B, bom equilíbrio qualidade/custo)
ollama run hf.co/mlx-community/gemma-4-26B-A4B-it-4bit

# MedGemma (fallback, fine-tune médico baseado no Gemma 3)
ollama run hf.co/mlx-community/medgemma-4b-it-6bit
```

> Para usar um modelo diferente do padrão, defina a variável de ambiente `OLLAMA_MODEL`:
> ```bash
> export OLLAMA_MODEL="hf.co/mlx-community/medgemma-4b-it-6bit"
> ```

---

## Como Executar

1. Inicie o servidor FastAPI (Backend dos Agentes):
```bash
python run_server.py
```

2. Em outro terminal, inicie a interface Streamlit:
```bash
streamlit run amie.py
```

3. Acesse `http://localhost:8501` no seu navegador.

---

## Referências

1. **Towards conversational diagnostic artificial intelligence** (Nature, 2025) — O paper original do AMIE.
2. **AMIE gains vision** (Google Research Blog, 2025) — Capacidades multimodais.
3. **From diagnosis to treatment** (Google Research Blog, 2025) — Arquitetura de dois agentes (Dialogue e Mx Agent).
4. **Towards physician-centered oversight of conversational diagnostic AI** (arXiv, 2025) — Framework de supervisão mdica e notas SOAP.
5. **MedGemma Model Card** (Google Health AI Developer Foundations) — Documentação oficial dos modelos.

---

## Migração do Gemma 3 / MedGemma

Se você usava o MedGemma (baseado no Gemma 3), as principais mudanças são:

| Antes | Agora |
|-------|-------|
| `medgemma-4b-it-6bit` (Ollama) | `gemma-4-E4B-it-6bit` (Ollama) |
| `gemma-3-27b-it` (Google AI) | `gemma-4-31b-it` (Google AI) |
| `num_ctx=32768` | `num_ctx=131072` (128K) |

Para continuar usando MedGemma localmente:
```bash
export OLLAMA_MODEL="hf.co/mlx-community/medgemma-4b-it-6bit"
```

---

## Aviso Legal

Este projeto é estritamente para fins de **pesquisa e desenvolvimento educacional**. Os modelos de IA (incluindo Gemma 4 e MedGemma) não são dispositivos médicos regulamentados e **não devem ser usados para diagnóstico, tratamento ou aconselhamento médico real** sem a supervisão de um profissional de saúde qualificado.
