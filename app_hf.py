import gradio as gr
import os
import logging
import asyncio
from datetime import date
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GOOGLE_API_KEY_ENV = os.environ.get("GOOGLE_API_KEY", "")

EXAMPLE = {
    "symptoms": "Dor toracica ha 2 horas, irradiando para braco esquerdo, sudorese e dispneia",
    "history": "HAS, DM2, tabagismo 20 anos",
    "medications": "Metformina 850mg 2x/dia, Losartana 50mg 1x/dia, AAS 100mg 1x/dia",
}

DIALOGUE_PROMPT = PromptTemplate(
    input_variables=["symptoms", "history"],
    template="""Voce e um medico assistente especializado em anamnese clinica.

Sintomas relatados: {symptoms}
Historico medico: {history}

Analise as informacoes e identifique **lacunas na anamnese**. Para cada lacuna:
- **Lacuna:** [informacao ausente]
  **Relevancia:** [por que e clinicamente importante]
  **Pergunta:** [pergunta direta ao paciente]

Ordene do mais ao menos urgente. Seja objetivo e focado.""",
)

CLINICAL_PROMPT = PromptTemplate(
    input_variables=["symptoms", "history"],
    template="""Voce e um especialista em raciocinio clinico e medicina baseada em evidencias.

Sintomas: {symptoms}
Historico medico: {history}

## Diagnosticos Diferenciais
Liste do mais ao menos provavel com justificativa breve.

## Exames Complementares
Priorize por urgencia (imediatos / em 24h / eletivos).

## Conduta Inicial
Medidas imediatas e plano de seguimento.

## Alertas Clinicos
Sinais de gravidade e criterios de internacao.

Base suas respostas em guidelines atuais (AHA, ESC, CFM).""",
)

MEDICATION_PROMPT = PromptTemplate(
    input_variables=["medications", "history"],
    template="""Voce e um especialista em farmacologia clinica e seguranca medicamentosa.

Medicacoes em uso: {medications}
Historico medico: {history}

## Interacoes Medicamentosas
Liste com nivel de gravidade: Leve / Moderada / Grave.

## Contraindicacoes
Com base no historico clinico.

## Ajustes de Dose
Se aplicavel, com justificativa.

## Monitoramento Recomendado
Parametros laboratoriais/clinicos e frequencia.""",
)


# -- Helpers -------------------------------------------------------------------

def get_llm(api_key: str, model: str = "gemma-4-31b-it"):
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.7,
    )


async def run_agent_async(prompt_template, inputs: dict, api_key: str, model: str = "gemma-4-31b-it") -> str:
    llm = get_llm(api_key, model=model)
    chain = prompt_template | llm
    response = await chain.ainvoke(inputs)
    return response.content


def transcribe_audio(audio_path, target_field, api_key, cur_symptoms, cur_history, cur_meds):
    if audio_path is None:
        return cur_symptoms, cur_history, cur_meds, "Nenhum audio gravado."

    effective_key = api_key.strip() or GOOGLE_API_KEY_ENV
    if not effective_key:
        return cur_symptoms, cur_history, cur_meds, "Informe sua API Key."

    try:
        import google.generativeai as genai
        genai.configure(api_key=effective_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        audio_file = genai.upload_file(path=audio_path)
        response = model.generate_content([
            "Transcreva este audio em portugues brasileiro. "
            "Retorne APENAS a transcricao, sem comentarios ou formatacao adicional.",
            audio_file,
        ])
        genai.delete_file(audio_file.name)
        text = response.text.strip()
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return cur_symptoms, cur_history, cur_meds, f"Erro: {str(e)}"

    def append(current, new):
        if current.strip():
            return current + "\n" + new
        return new

    if target_field == "Sintomas":
        return append(cur_symptoms, text), cur_history, cur_meds, text
    elif target_field == "Historico":
        return cur_symptoms, append(cur_history, text), cur_meds, text
    else:
        return cur_symptoms, cur_history, append(cur_meds, text), text


def _pipeline_html(states=None):
    if states is None:
        states = ["pending", "pending", "pending"]
    agents = [
        ("Anamnese", "Lacunas clinicas"),
        ("Diagnostico", "Raciocinio clinico"),
        ("Farmacia", "Seguranca meds."),
    ]
    icons = {
        "pending": '<span class="pip-icon pip-pending"></span>',
        "running": '<span class="pip-icon pip-running"></span>',
        "done": '<span class="pip-icon pip-done"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg></span>',
        "error": '<span class="pip-icon pip-error">!</span>',
    }
    steps = []
    for i, (name, desc) in enumerate(agents):
        state = states[i]
        steps.append(f"""
        <div class="pip-step pip-{state}">
            {icons[state]}
            <div class="pip-text">
                <div class="pip-name">{name}</div>
                <div class="pip-desc">{desc}</div>
            </div>
        </div>
        """)
        if i < 2:
            steps.append('<div class="pip-connector"></div>')
    return f'<div class="pipeline">{" ".join(steps)}</div>'


def analyze(symptoms, history, medications, api_key, model_name, progress=gr.Progress()):
    effective_key = api_key.strip() or GOOGLE_API_KEY_ENV
    if not effective_key:
        msg = "Informe sua Google API Key."
        return _pipeline_html(["error", "error", "error"]), msg, msg, msg
    if not all([symptoms.strip(), history.strip(), medications.strip()]):
        msg = "Preencha todos os campos."
        return _pipeline_html(["error", "error", "error"]), msg, msg, msg

    progress(0.1, desc="Executando 3 agentes em paralelo...")

    async def run_all():
        return await asyncio.gather(
            run_agent_async(DIALOGUE_PROMPT, {"symptoms": symptoms, "history": history}, effective_key, model=model_name),
            run_agent_async(CLINICAL_PROMPT, {"symptoms": symptoms, "history": history}, effective_key, model=model_name),
            run_agent_async(MEDICATION_PROMPT, {"medications": medications, "history": history}, effective_key, model=model_name),
            return_exceptions=True,
        )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        progress(0.4, desc="Aguardando respostas...")
        results = loop.run_until_complete(run_all())
        loop.close()
        progress(1.0, desc="Concluido.")
    except Exception as e:
        err = f"Erro: {str(e)}"
        return _pipeline_html(["error", "error", "error"]), err, err, err

    processed = []
    states = []
    for r in results:
        if isinstance(r, Exception):
            processed.append(f"Erro neste agente: {str(r)}")
            states.append("error")
        else:
            processed.append(r)
            states.append("done")

    return _pipeline_html(states), processed[0], processed[1], processed[2]


def export_soap(symptoms, history, medications, dialogue, clinical, medication):
    if not any([dialogue.strip(), clinical.strip(), medication.strip()]):
        return gr.update(value="Execute a analise primeiro.", visible=True)

    soap = f"""# Evolucao Medica -- SOAP
**Data:** {date.today().strftime('%d/%m/%Y')}

---

## S -- Subjetivo
**Sintomas:** {symptoms}
**Historico:** {history}
**Medicacoes:** {medications}

---

## O -- Objetivo
*[Preencher com exame fisico e sinais vitais]*

---

## A -- Avaliacao

### Anamnese
{dialogue}

### Raciocinio Clinico
{clinical}

---

## P -- Plano

### Seguranca Medicamentosa
{medication}

---
*AMIE Medical Agents -- Revisao obrigatoria por profissional habilitado*"""

    return gr.update(value=soap, visible=True)


def export_markdown(symptoms, history, medications, dialogue, clinical, medication):
    if not any([dialogue.strip(), clinical.strip(), medication.strip()]):
        return gr.update(value="Execute a analise primeiro.", visible=True)

    md = f"""# Relatorio Clinico -- AMIE Medical Agents
**Data:** {date.today().strftime('%d/%m/%Y')}

---

## Dados do Paciente
- **Sintomas:** {symptoms}
- **Historico:** {history}
- **Medicacoes:** {medications}

---

## Anamnese -- Lacunas Identificadas
{dialogue}

---

## Raciocinio Clinico
{clinical}

---

## Seguranca Medicamentosa
{medication}

---
*Gerado por AMIE Medical Agents -- Revisao obrigatoria por profissional habilitado*"""

    return gr.update(value=md, visible=True)


def load_example():
    return EXAMPLE["symptoms"], EXAMPLE["history"], EXAMPLE["medications"]


# -- CSS -----------------------------------------------------------------------
TEAL = "#0f6e56"
TEAL_LIGHT = "#e6f5f0"
TEAL_HOVER = "#0b5a46"

CSS = """
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,400;0,500;0,600;0,700;1,400&family=DM+Mono:wght@400;500&display=swap');

:root, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    --body-background-fill: #f7f8fa !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e4e7ec !important;
    --block-label-background-fill: #ffffff !important;
    --block-label-text-color: #555555 !important;
    --block-title-text-color: #1a1a1a !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #dde1e6 !important;
    --input-placeholder-color: #a0a7b0 !important;
    --panel-background-fill: #ffffff !important;
    --panel-border-color: #e4e7ec !important;
    --background-fill-primary: #f7f8fa !important;
    --background-fill-secondary: #ffffff !important;
    --border-color-primary: #e4e7ec !important;
    --border-color-accent: """ + TEAL + """ !important;
    --color-accent: """ + TEAL + """ !important;
    --body-text-color: #1a1a1a !important;
    --body-text-color-subdued: #6b7280 !important;
    --link-text-color: """ + TEAL + """ !important;
    --button-primary-background-fill: """ + TEAL + """ !important;
    --button-primary-background-fill-hover: """ + TEAL_HOVER + """ !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: """ + TEAL + """ !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: """ + TEAL_LIGHT + """ !important;
    --button-secondary-text-color: """ + TEAL + """ !important;
    --button-secondary-border-color: #d0d5dd !important;
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .gradio-container, .main, footer {
    font-family: 'DM Sans', system-ui, -apple-system, sans-serif !important;
    background: #f7f8fa !important;
    color: #1a1a1a !important;
}

footer { display: none !important; }
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; padding: 0 16px !important; }

/* -- Typography -------------------------------------------------------------- */
textarea, input[type="text"], input[type="password"] {
    background: #fff !important;
    border: 1px solid #dde1e6 !important;
    border-radius: 10px !important;
    color: #1a1a1a !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.2s !important;
    box-shadow: none !important;
}

textarea:focus, input:focus {
    border-color: """ + TEAL + """ !important;
    box-shadow: 0 0 0 3px rgba(15,110,86,0.08) !important;
    outline: none !important;
}

label > span, .label-wrap span {
    font-size: 11px !important;
    font-weight: 500 !important;
    color: #6b7280 !important;
    font-family: 'DM Mono', monospace !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* -- Blocks ------------------------------------------------------------------ */
.block, .form, .gap, div[class*="block"] {
    background: transparent !important;
    box-shadow: none !important;
}

/* -- Panels ------------------------------------------------------------------ */
.input-panel {
    background: #ffffff !important;
    border: 1px solid #e4e7ec !important;
    border-radius: 16px !important;
    padding: 0 !important;
}

.output-panel {
    background: transparent !important;
    padding: 0 !important;
}

/* -- Buttons ----------------------------------------------------------------- */
button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.2s !important;
    box-shadow: none !important;
}

button.primary, button[variant="primary"] {
    border-radius: 12px !important;
}

button.secondary, button[variant="secondary"] {
    border-radius: 100px !important;
    background: #fff !important;
    color: """ + TEAL + """ !important;
    border: 1px solid #d0d5dd !important;
    font-size: 13px !important;
}

button.secondary:hover { background: """ + TEAL_LIGHT + """ !important; border-color: """ + TEAL + """ !important; }
button.sm { padding: 6px 16px !important; font-size: 12px !important; }

/* -- Analyze button --------------------------------------------------------- */
.analyze-btn button {
    width: 100% !important;
    padding: 14px 24px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 12px !important;
    background: """ + TEAL + """ !important;
    color: #fff !important;
    border: none !important;
    min-height: 48px !important;
    letter-spacing: -0.2px !important;
}

.analyze-btn button:hover {
    background: """ + TEAL_HOVER + """ !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(15,110,86,0.2) !important;
}

/* -- Pipeline bar ----------------------------------------------------------- */
.pipeline {
    display: flex;
    align-items: center;
    gap: 0;
    padding: 14px 18px;
    background: #fff;
    border: 1px solid #e4e7ec;
    border-radius: 14px;
    margin-bottom: 12px;
    font-family: 'DM Sans', sans-serif;
}

.pip-step {
    display: flex;
    align-items: center;
    gap: 8px;
    flex: 1;
}

.pip-connector {
    width: 24px;
    height: 2px;
    background: #e4e7ec;
    flex-shrink: 0;
    margin: 0 4px;
}

.pip-icon {
    width: 24px; height: 24px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 11px; font-weight: 700;
    transition: all 0.3s;
}

.pip-pending .pip-icon { background: #f3f4f6; border: 2px solid #d1d5db; }
.pip-running .pip-icon { background: #fff; border: 2px solid """ + TEAL + """; animation: pip-spin 1s linear infinite; }
.pip-done .pip-icon { background: """ + TEAL + """; border: 2px solid """ + TEAL + """; }
.pip-error .pip-icon { background: #fef2f2; border: 2px solid #ef4444; color: #ef4444; }

@keyframes pip-spin {
    0% { box-shadow: 0 0 0 0 rgba(15,110,86,0.3); }
    100% { box-shadow: 0 0 0 6px rgba(15,110,86,0); }
}

.pip-name {
    font-size: 12px; font-weight: 600; color: #1a1a1a;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.2;
}

.pip-desc {
    font-size: 10px; color: #9ca3af;
    font-family: 'DM Mono', monospace;
    line-height: 1.2;
}

.pip-done .pip-name { color: """ + TEAL + """; }

/* -- Result cards ----------------------------------------------------------- */
.result-card {
    background: #fff !important;
    border: 1px solid #e4e7ec !important;
    border-radius: 14px !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}

.result-header {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 16px;
    font-family: 'DM Sans', sans-serif;
    font-size: 13px;
    font-weight: 600;
    border-bottom: 1px solid #f0f2f5;
}

.result-header .rh-icon {
    width: 28px; height: 28px;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
    font-size: 14px;
}

.result-header .rh-label {
    font-family: 'DM Mono', monospace;
    font-size: 10px;
    font-weight: 400;
    color: #9ca3af;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

.rh-anamnese { background: #e6f5f0; color: #0f6e56; }
.rh-anamnese .rh-icon { background: #d1ede4; color: #0f6e56; }

.rh-clinical { background: #e8f4fd; color: #1a5276; }
.rh-clinical .rh-icon { background: #d4e8f7; color: #1a5276; }

.rh-safety { background: #fef7e6; color: #92400e; }
.rh-safety .rh-icon { background: #fdecc8; color: #92400e; }

.result-body {
    padding: 14px 16px !important;
}

.result-body .prose, .result-body .markdown {
    color: #374151 !important;
    font-size: 13px !important;
    line-height: 1.75 !important;
}

.result-body .prose h2, .result-body .markdown h2 {
    font-size: 13px !important; font-weight: 600 !important;
    color: #1a1a1a !important; margin: 16px 0 6px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid #f3f4f6 !important;
    font-family: 'DM Sans', sans-serif !important;
}

.result-body .prose strong, .result-body .markdown strong { color: #1a1a1a !important; }

/* -- Export footer ---------------------------------------------------------- */
.export-footer {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 12px 16px;
    background: #f9fafb;
    border-top: 1px solid #f0f2f5;
    border-radius: 0 0 14px 14px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #9ca3af;
}

/* -- Model selector compact ------------------------------------------------- */
.model-compact {
    background: transparent !important;
    border: none !important;
}

.model-compact .block {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.model-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 12px;
    background: #f3f4f6;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #6b7280;
    cursor: default;
}

.model-tag .dot {
    width: 6px; height: 6px;
    background: """ + TEAL + """;
    border-radius: 50%;
}

/* -- Voice compact ---------------------------------------------------------- */
.voice-compact {
    background: #f9fafb !important;
    border: 1px solid #e4e7ec !important;
    border-radius: 12px !important;
    padding: 12px !important;
    margin-bottom: 4px !important;
}

.voice-tag {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 8px;
}

.voice-tag svg { opacity: 0.5; }

/* -- Accordion -------------------------------------------------------------- */
details, details summary {
    background: #fff !important;
    border: 1px solid #e4e7ec !important;
    border-radius: 12px !important;
    color: #1a1a1a !important;
}

details[open] { border-color: #d0d5dd !important; }

details summary {
    font-size: 12px !important; font-weight: 500 !important;
    padding: 8px 14px !important; border: none !important;
    color: #9ca3af !important;
    font-family: 'DM Mono', monospace !important;
}

/* -- Disclaimer ------------------------------------------------------------- */
.disclaimer {
    font-size: 11px; color: #b0b7c3;
    border-top: 1px solid #f0f2f5;
    padding-top: 14px; margin-top: 18px;
    font-family: 'DM Sans', sans-serif;
    line-height: 1.6;
    text-align: center;
}

.disclaimer a { color: """ + TEAL + """; text-decoration: none; }

/* -- Header ----------------------------------------------------------------- */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 0 14px;
    border-bottom: 1px solid #e4e7ec;
    margin-bottom: 16px;
    font-family: 'DM Sans', sans-serif;
}

.app-header-left {
    display: flex;
    align-items: center;
    gap: 12px;
}

.app-logo {
    width: 34px; height: 34px;
    background: """ + TEAL + """;
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 15px; font-weight: 700;
    font-family: 'DM Sans', sans-serif;
    flex-shrink: 0;
}

.app-title {
    font-size: 16px; font-weight: 700; color: #1a1a1a;
    line-height: 1.2; letter-spacing: -0.3px;
}

.app-subtitle {
    font-size: 11px; color: #9ca3af;
    font-family: 'DM Mono', monospace;
    margin-top: 2px;
}

.app-badges {
    display: flex;
    align-items: center;
    gap: 8px;
}

.badge {
    padding: 4px 10px;
    border-radius: 100px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'DM Mono', monospace;
    letter-spacing: 0.3px;
}

.badge-online { background: """ + TEAL_LIGHT + """; color: """ + TEAL + """; }
.badge-lang { background: #f3f4f6; color: #6b7280; }

/* -- Mic button ------------------------------------------------------------- */
.mic-btn {
    position: absolute;
    bottom: 8px; right: 8px;
    width: 30px; height: 30px;
    background: #fff;
    border: 1px solid #dde1e6 !important;
    border-radius: 8px !important;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #9ca3af;
    padding: 0 !important;
    transition: all 0.2s !important;
    z-index: 10;
    box-shadow: 0 1px 2px rgba(0,0,0,0.05) !important;
}

.mic-btn:hover { border-color: """ + TEAL + """ !important; color: """ + TEAL + """; background: """ + TEAL_LIGHT + """; }

.mic-btn.recording {
    background: """ + TEAL + """ !important;
    border-color: """ + TEAL + """ !important;
    color: #fff !important;
    animation: mic-pulse 1.2s ease-in-out infinite;
}

@keyframes mic-pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(15,110,86,0.3) !important; }
    50% { box-shadow: 0 0 0 5px rgba(15,110,86,0.08) !important; }
}

/* -- Transcription status --------------------------------------------------- */
.transcription-status {
    font-size: 12px;
    color: #9ca3af;
    font-family: 'DM Mono', monospace;
    padding: 4px 0;
    min-height: 20px;
}

/* -- Mobile ----------------------------------------------------------------- */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; max-width: 100% !important; }
    textarea { font-size: 16px !important; }
    .analyze-btn button { font-size: 16px !important; min-height: 52px !important; }
    .pipeline { flex-direction: column; gap: 8px; align-items: flex-start; }
    .pip-connector { width: 2px; height: 16px; margin: 0 0 0 11px; }
    .app-badges { display: none; }
    .app-header { padding: 12px 0 10px; }
}
"""

# -- JS (mic injection) -------------------------------------------------------
JS = """
() => {
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const MIC_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/><line x1="8" y1="23" x2="16" y2="23"/></svg>`;
    const STOP_SVG = `<svg width="14" height="14" viewBox="0 0 24 24" fill="currentColor"><rect x="4" y="4" width="16" height="16" rx="2"/></svg>`;

    function injectMic(textarea) {
        if (textarea.dataset.micAdded) return;
        textarea.dataset.micAdded = '1';
        const wrap = textarea.closest('.block') || textarea.parentElement;
        if (!wrap) return;
        if (window.getComputedStyle(wrap).position === 'static') wrap.style.position = 'relative';

        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = 'mic-btn';
        btn.title = SR ? 'Ditar (pt-BR)' : 'Navegador nao suporta ditado';
        btn.innerHTML = MIC_SVG;
        if (!SR) { btn.style.opacity = '0.3'; btn.style.cursor = 'not-allowed'; }
        wrap.appendChild(btn);
        if (!SR) return;

        let rec = null, active = false;
        btn.addEventListener('click', (e) => {
            e.preventDefault(); e.stopPropagation();
            if (active) { rec && rec.stop(); return; }
            rec = new SR();
            rec.lang = 'pt-BR';
            rec.continuous = true;
            rec.interimResults = false;

            rec.onstart = () => { active = true; btn.classList.add('recording'); btn.title = 'Parar'; btn.innerHTML = STOP_SVG; };
            rec.onresult = (ev) => {
                let text = '';
                for (let i = ev.resultIndex; i < ev.results.length; i++) {
                    if (ev.results[i].isFinal) text += ev.results[i][0].transcript;
                }
                if (!text) return;
                const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set;
                const cur = textarea.value;
                setter.call(textarea, cur + (cur.trim() ? ' ' : '') + text.trim());
                textarea.dispatchEvent(new Event('input', { bubbles: true }));
                textarea.dispatchEvent(new Event('change', { bubbles: true }));
            };
            rec.onerror = () => rec.stop();
            rec.onend = () => { active = false; btn.classList.remove('recording'); btn.title = 'Ditar (pt-BR)'; btn.innerHTML = MIC_SVG; };
            rec.start();
        });
    }

    function scan() { document.querySelectorAll('textarea:not([data-mic-added])').forEach(injectMic); }
    new MutationObserver(scan).observe(document.body, { childList: true, subtree: true });
    setTimeout(scan, 600);
}
"""

# -- App -----------------------------------------------------------------------

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.emerald,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("DM Sans"),
        font_mono=gr.themes.GoogleFont("DM Mono"),
    ).set(
        body_background_fill="#f7f8fa",
        body_background_fill_dark="#f7f8fa",
        body_text_color="#1a1a1a",
        body_text_color_dark="#1a1a1a",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#e4e7ec",
        block_border_color_dark="#e4e7ec",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#dde1e6",
        input_border_color_dark="#dde1e6",
        button_primary_background_fill=TEAL,
        button_primary_background_fill_dark=TEAL,
        button_primary_background_fill_hover=TEAL_HOVER,
        button_primary_background_fill_hover_dark=TEAL_HOVER,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color=TEAL,
        button_secondary_text_color_dark=TEAL,
        button_secondary_border_color="#d0d5dd",
        button_secondary_border_color_dark="#d0d5dd",
        background_fill_primary="#f7f8fa",
        background_fill_primary_dark="#f7f8fa",
        background_fill_secondary="#ffffff",
        background_fill_secondary_dark="#ffffff",
    ),
    css=CSS,
    title="AMIE Medical Agents",
    js=JS,
) as demo:

    # -- Header ----------------------------------------------------------------
    gr.HTML("""
    <div class="app-header">
        <div class="app-header-left">
            <div class="app-logo">A</div>
            <div>
                <div class="app-title">AMIE Medical Agents</div>
                <div class="app-subtitle">3 agentes clinicos &middot; gemma-4-31b-it</div>
            </div>
        </div>
        <div class="app-badges">
            <span class="badge badge-online">online</span>
            <span class="badge badge-lang">pt-BR</span>
        </div>
    </div>
    """)

    # -- Two-column layout -----------------------------------------------------
    with gr.Row(equal_height=False):

        # ===== LEFT: Input panel ==============================================
        with gr.Column(scale=2, elem_classes=["input-panel"]):

            # -- Voice (compact) -----------------------------------------------
            with gr.Group(elem_classes=["voice-compact"]):
                gr.HTML("""
                <div class="voice-tag">
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                        <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>
                        <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                    </svg>
                    Gravacao de voz &mdash; transcricao automatica via Gemini Flash
                </div>
                """)
                with gr.Row():
                    audio_input = gr.Audio(
                        sources=["microphone", "upload"],
                        type="filepath",
                        label="Audio",
                        scale=3,
                    )
                    target_field = gr.Dropdown(
                        choices=["Sintomas", "Historico", "Medicacoes"],
                        value="Sintomas",
                        label="Preencher campo",
                        scale=1,
                    )
                transcribe_btn = gr.Button("Transcrever e preencher", variant="secondary", size="sm")
                transcription_status = gr.Markdown("", elem_classes=["transcription-status"])

            # -- Patient data --------------------------------------------------
            gr.HTML("""
            <div style="display:flex;align-items:center;justify-content:space-between;padding:4px 4px 0">
                <div style="display:flex;align-items:center;gap:6px">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#6b7280" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="8.5" cy="7" r="4"/></svg>
                    <span style="font-family:'DM Mono',monospace;font-size:11px;color:#6b7280;text-transform:uppercase;letter-spacing:0.8px">Dados do paciente</span>
                </div>
            </div>
            """)

            symptoms = gr.Textbox(label="SINTOMAS", placeholder="Queixa principal, inicio, caracterizacao...", lines=2)
            history = gr.Textbox(label="HISTORICO MEDICO", placeholder="Comorbidades, cirurgias, alergias...", lines=2)
            medications = gr.Textbox(label="MEDICACOES", placeholder="Nome, dose e frequencia...", lines=2)

            with gr.Row():
                example_btn = gr.Button("Carregar exemplo", variant="secondary", size="sm")

            # -- Analyze button ------------------------------------------------
            run_btn = gr.Button("Analisar com 3 agentes", variant="primary", size="lg", elem_classes=["analyze-btn"])

            # -- Model selector (compact inline) -------------------------------
            with gr.Row():
                gr.HTML('<div class="model-tag"><span class="dot"></span>Modelo ativo</div>')
                model_selector = gr.Dropdown(
                    choices=[
                        "gemma-4-31b-it",
                        "gemma-4-26b-a4b-it",
                        "gemma-3-27b-it",
                    ],
                    value="gemma-4-31b-it",
                    label="",
                    show_label=False,
                    scale=2,
                    elem_classes=["model-compact"],
                )

            # -- API Key (discrete) --------------------------------------------
            with gr.Accordion("API Key", open=not bool(GOOGLE_API_KEY_ENV)):
                api_key_input = gr.Textbox(
                    label="Google API Key",
                    placeholder="Obtenha em aistudio.google.com",
                    type="password",
                    value=GOOGLE_API_KEY_ENV,
                )

        # ===== RIGHT: Output panel ============================================
        with gr.Column(scale=3, elem_classes=["output-panel"]):

            # -- Pipeline bar --------------------------------------------------
            pipeline_bar = gr.HTML(value=_pipeline_html(), elem_classes=["pipeline-wrap"])

            # -- Anamnese card -------------------------------------------------
            gr.HTML("""
            <div class="result-card">
                <div class="result-header rh-anamnese">
                    <div class="rh-icon">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/></svg>
                    </div>
                    <div>
                        <div>Anamnese &mdash; lacunas</div>
                        <div class="rh-label">Agente de dialogo</div>
                    </div>
                </div>
            </div>
            """)
            out_dialogue = gr.Markdown(value="", elem_classes=["result-body"])

            # -- Clinical card -------------------------------------------------
            gr.HTML("""
            <div class="result-card">
                <div class="result-header rh-clinical">
                    <div class="rh-icon">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
                    </div>
                    <div>
                        <div>Raciocinio clinico</div>
                        <div class="rh-label">Diagnostico + conduta</div>
                    </div>
                </div>
            </div>
            """)
            out_clinical = gr.Markdown(value="", elem_classes=["result-body"])

            # -- Safety card ---------------------------------------------------
            gr.HTML("""
            <div class="result-card">
                <div class="result-header rh-safety">
                    <div class="rh-icon">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
                    </div>
                    <div>
                        <div>Seguranca farmacologica</div>
                        <div class="rh-label">Interacoes + ajustes</div>
                    </div>
                </div>
            </div>
            """)
            out_medication = gr.Markdown(value="", elem_classes=["result-body"])

            # -- Export footer -------------------------------------------------
            gr.HTML('<div class="export-footer">Exportar evolucao</div>')
            with gr.Row():
                export_soap_btn = gr.Button("SOAP", variant="secondary", size="sm")
                export_md_btn = gr.Button("Markdown", variant="secondary", size="sm")
            soap_out = gr.Markdown(visible=False)

    # -- Disclaimer ------------------------------------------------------------
    gr.HTML("""
    <div class="disclaimer">
        Ferramenta de suporte clinico &mdash; nao substitui o julgamento medico.
        Toda conduta deve ser validada por profissional habilitado.<br/>
        Inspirado no AMIE (Google DeepMind) &mdash; Powered by Gemma 4 via Google AI.
    </div>
    """)

    # -- Events ----------------------------------------------------------------
    transcribe_btn.click(
        fn=transcribe_audio,
        inputs=[audio_input, target_field, api_key_input, symptoms, history, medications],
        outputs=[symptoms, history, medications, transcription_status],
    )

    run_btn.click(
        fn=analyze,
        inputs=[symptoms, history, medications, api_key_input, model_selector],
        outputs=[pipeline_bar, out_dialogue, out_clinical, out_medication],
    )

    example_btn.click(
        fn=load_example,
        outputs=[symptoms, history, medications],
    )

    export_soap_btn.click(
        fn=export_soap,
        inputs=[symptoms, history, medications, out_dialogue, out_clinical, out_medication],
        outputs=[soap_out],
    )

    export_md_btn.click(
        fn=export_markdown,
        inputs=[symptoms, history, medications, out_dialogue, out_clinical, out_medication],
        outputs=[soap_out],
    )

if __name__ == "__main__":
    demo.launch()
