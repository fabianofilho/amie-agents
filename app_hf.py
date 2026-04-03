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
    """Transcribe audio using Gemini Flash and route to selected field."""
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


def analyze(symptoms, history, medications, api_key, model_name, progress=gr.Progress()):
    effective_key = api_key.strip() or GOOGLE_API_KEY_ENV
    if not effective_key:
        msg = "Informe sua Google API Key."
        return msg, msg, msg
    if not all([symptoms.strip(), history.strip(), medications.strip()]):
        msg = "Preencha todos os campos."
        return msg, msg, msg

    progress(0.1, desc="Agente 1/3: Anamnese...")

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
        progress(0.3, desc="Agente 2/3: Raciocinio clinico...")
        results = loop.run_until_complete(run_all())
        loop.close()
        progress(1.0, desc="Concluido.")
    except Exception as e:
        err = f"Erro: {str(e)}"
        return err, err, err

    processed = []
    for r in results:
        if isinstance(r, Exception):
            processed.append(f"Erro neste agente: {str(r)}")
        else:
            processed.append(r)

    return processed[0], processed[1], processed[2]


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


def load_example():
    return EXAMPLE["symptoms"], EXAMPLE["history"], EXAMPLE["medications"]


# -- CSS -----------------------------------------------------------------------
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root, .dark, [data-theme="dark"] {
    color-scheme: light !important;
    --body-background-fill: #ffffff !important;
    --block-background-fill: #ffffff !important;
    --block-border-color: #e8e8e8 !important;
    --block-label-background-fill: #ffffff !important;
    --block-label-text-color: #555555 !important;
    --block-title-text-color: #111111 !important;
    --input-background-fill: #ffffff !important;
    --input-border-color: #e0e0e0 !important;
    --input-placeholder-color: #b0b0b0 !important;
    --panel-background-fill: #fafafa !important;
    --panel-border-color: #e8e8e8 !important;
    --background-fill-primary: #ffffff !important;
    --background-fill-secondary: #fafafa !important;
    --border-color-primary: #e8e8e8 !important;
    --border-color-accent: #111111 !important;
    --color-accent: #111111 !important;
    --body-text-color: #111111 !important;
    --body-text-color-subdued: #666666 !important;
    --link-text-color: #111111 !important;
    --button-primary-background-fill: #111111 !important;
    --button-primary-background-fill-hover: #333333 !important;
    --button-primary-text-color: #ffffff !important;
    --button-primary-border-color: #111111 !important;
    --button-secondary-background-fill: #ffffff !important;
    --button-secondary-background-fill-hover: #f5f5f5 !important;
    --button-secondary-text-color: #333333 !important;
    --button-secondary-border-color: #dddddd !important;
    --shadow-drop: none !important;
    --shadow-drop-lg: none !important;
}

*, *::before, *::after { box-sizing: border-box; }

html, body, .gradio-container, .main, footer {
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
    background: #ffffff !important;
    color: #111111 !important;
}

footer { display: none !important; }
.gradio-container { max-width: 720px !important; margin: 0 auto !important; }

/* -- Inputs ----------------------------------------------------------------- */
textarea, input[type="text"], input[type="password"] {
    background: #fff !important;
    border: 1px solid #e0e0e0 !important;
    border-radius: 8px !important;
    color: #111 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    transition: border-color 0.15s !important;
    box-shadow: none !important;
}

textarea:focus, input:focus {
    border-color: #111 !important;
    box-shadow: none !important;
    outline: none !important;
}

label > span, .label-wrap span {
    font-size: 12px !important;
    font-weight: 600 !important;
    color: #555 !important;
    font-family: 'Inter', sans-serif !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

/* -- Blocks flat ------------------------------------------------------------- */
.block, .form, .gap, div[class*="block"] {
    background: #fff !important;
    box-shadow: none !important;
}

/* -- Buttons ----------------------------------------------------------------- */
button {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    transition: all 0.15s !important;
    box-shadow: none !important;
    letter-spacing: 0 !important;
}

button.primary, button[variant="primary"] {
    border-radius: 10px !important;
}

button.secondary, button[variant="secondary"] {
    border-radius: 100px !important;
    background: #fff !important;
    color: #444 !important;
    border: 1px solid #ddd !important;
}

button.secondary:hover { background: #f5f5f5 !important; border-color: #bbb !important; }
button.sm { padding: 6px 14px !important; font-size: 12px !important; }

/* -- Voice section ---------------------------------------------------------- */
.voice-section {
    background: #f8f9fa !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 12px !important;
    padding: 16px !important;
    margin-bottom: 12px !important;
}

.voice-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-family: 'Inter', sans-serif;
}

.voice-icon {
    width: 32px; height: 32px;
    background: #111;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
}

.voice-label {
    font-size: 13px; font-weight: 600; color: #111;
}

.voice-sublabel {
    font-size: 11px; color: #999;
}

/* -- Analyze button --------------------------------------------------------- */
.analyze-btn button {
    width: 100% !important;
    padding: 14px 24px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    background: #111 !important;
    color: #fff !important;
    border: none !important;
    min-height: 48px !important;
}

.analyze-btn button:hover {
    background: #333 !important;
}

/* -- Result cards ----------------------------------------------------------- */
.result-card {
    border: 1px solid #e8e8e8 !important;
    border-radius: 10px !important;
    overflow: hidden !important;
    margin-bottom: 10px !important;
}

.result-header-anamnese {
    background: #EBF5FB !important;
    border-bottom: 1px solid #d4e6f1 !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #1a5276 !important;
    font-family: 'Inter', sans-serif !important;
}

.result-header-clinical {
    background: #E8F8F5 !important;
    border-bottom: 1px solid #d1f2eb !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #0e6655 !important;
    font-family: 'Inter', sans-serif !important;
}

.result-header-safety {
    background: #FEF5E7 !important;
    border-bottom: 1px solid #fdebd0 !important;
    padding: 10px 14px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
    color: #935116 !important;
    font-family: 'Inter', sans-serif !important;
}

.result-body {
    padding: 12px 14px !important;
}

/* -- Markdown in results ---------------------------------------------------- */
.prose, .markdown {
    color: #222 !important;
    font-size: 13px !important;
    line-height: 1.7 !important;
}

.prose h2, .markdown h2 {
    font-size: 13px !important; font-weight: 600 !important;
    color: #111 !important; margin: 14px 0 6px !important;
    padding-bottom: 4px !important;
    border-bottom: 1px solid #f0f0f0 !important;
}

.prose strong, .markdown strong { color: #111 !important; }

/* -- Accordion -------------------------------------------------------------- */
details, details summary {
    background: #fff !important;
    border: 1px solid #e8e8e8 !important;
    border-radius: 10px !important;
    color: #111 !important;
}

details[open] { border-color: #ccc !important; }

details summary {
    font-size: 12px !important; font-weight: 500 !important;
    padding: 8px 14px !important; border: none !important;
    color: #888 !important;
}

/* -- Disclaimer ------------------------------------------------------------- */
.disclaimer {
    font-size: 11px; color: #bbb;
    border-top: 1px solid #f0f0f0;
    padding-top: 12px; margin-top: 16px;
    font-family: 'Inter', sans-serif;
    line-height: 1.5;
    text-align: center;
}

/* -- Mic button (JS inline) ------------------------------------------------- */
.mic-btn {
    position: absolute;
    bottom: 8px; right: 8px;
    width: 32px; height: 32px;
    background: #fff;
    border: 1px solid #ddd !important;
    border-radius: 50% !important;
    cursor: pointer;
    display: flex; align-items: center; justify-content: center;
    color: #888;
    padding: 0 !important;
    transition: all 0.2s !important;
    z-index: 10;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
.mic-btn:hover { border-color: #111 !important; color: #111; }
.mic-btn.recording {
    background: #111 !important;
    border-color: #111 !important;
    color: #fff !important;
    animation: pulse 1s infinite;
}
@keyframes pulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(0,0,0,0.3) !important; }
    50% { box-shadow: 0 0 0 4px rgba(0,0,0,0.1) !important; }
}

/* -- Transcription status --------------------------------------------------- */
.transcription-status {
    font-size: 12px;
    color: #888;
    font-family: 'Inter', sans-serif;
    padding: 4px 0;
    min-height: 20px;
}

.transcription-status.success { color: #0e6655; }
.transcription-status.error { color: #c0392b; }

/* -- Mobile ----------------------------------------------------------------- */
@media (max-width: 768px) {
    .gradio-container { padding: 8px !important; }
    textarea { font-size: 16px !important; }
    .analyze-btn button { font-size: 16px !important; min-height: 52px !important; }
}
"""

with gr.Blocks(
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.neutral,
        secondary_hue=gr.themes.colors.neutral,
        neutral_hue=gr.themes.colors.neutral,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ).set(
        body_background_fill="#ffffff",
        body_background_fill_dark="#ffffff",
        body_text_color="#111111",
        body_text_color_dark="#111111",
        block_background_fill="#ffffff",
        block_background_fill_dark="#ffffff",
        block_border_color="#e8e8e8",
        block_border_color_dark="#e8e8e8",
        input_background_fill="#ffffff",
        input_background_fill_dark="#ffffff",
        input_border_color="#e0e0e0",
        input_border_color_dark="#e0e0e0",
        button_primary_background_fill="#111111",
        button_primary_background_fill_dark="#111111",
        button_primary_background_fill_hover="#333333",
        button_primary_background_fill_hover_dark="#333333",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#ffffff",
        button_secondary_background_fill_dark="#ffffff",
        button_secondary_text_color="#333333",
        button_secondary_text_color_dark="#333333",
        button_secondary_border_color="#dddddd",
        button_secondary_border_color_dark="#dddddd",
        background_fill_primary="#ffffff",
        background_fill_primary_dark="#ffffff",
        background_fill_secondary="#fafafa",
        background_fill_secondary_dark="#fafafa",
    ),
    css=CSS,
    title="AMIE Medical Agents",
    js="""
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
""",
) as demo:

    # -- Header ----------------------------------------------------------------
    gr.HTML("""
    <div style="display:flex;align-items:center;gap:10px;padding:16px 0 14px;border-bottom:1px solid #ebebeb;margin-bottom:16px">
        <div style="width:30px;height:30px;background:#111;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#fff;font-size:14px;font-weight:700;font-family:Inter,sans-serif;flex-shrink:0">A</div>
        <div>
            <div style="font-size:15px;font-weight:600;color:#111;font-family:Inter,sans-serif;line-height:1.2">AMIE Medical Agents</div>
            <div style="font-size:11px;color:#999;font-family:Inter,sans-serif;margin-top:1px">3 agentes clinicos -- Powered by Gemma 4</div>
        </div>
    </div>
    """)

    # -- Voice section (always visible) ----------------------------------------
    gr.HTML("""
    <div class="voice-header">
        <div class="voice-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
                <path d="M12 2a3 3 0 0 1 3 3v7a3 3 0 0 1-6 0V5a3 3 0 0 1 3-3z"/>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"/>
                <line x1="12" y1="19" x2="12" y2="23"/>
                <line x1="8" y1="23" x2="16" y2="23"/>
            </svg>
        </div>
        <div>
            <div class="voice-label">Gravacao de voz</div>
            <div class="voice-sublabel">Grave a consulta e a transcricao preenche o campo escolhido automaticamente</div>
        </div>
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

    # -- Input fields ----------------------------------------------------------
    symptoms = gr.Textbox(label="Sintomas", placeholder="Queixa principal, inicio, caracterizacao...", lines=2)
    history = gr.Textbox(label="Historico", placeholder="Comorbidades, cirurgias, alergias...", lines=2)
    medications = gr.Textbox(label="Medicacoes", placeholder="Nome, dose e frequencia...", lines=2)

    with gr.Row():
        example_btn = gr.Button("Carregar exemplo", variant="secondary", size="sm")

    # -- Analyze button (prominent) --------------------------------------------
    run_btn = gr.Button("Analisar", variant="primary", size="lg", elem_classes=["analyze-btn"])

    # -- Results (all visible, no tabs) ----------------------------------------
    gr.HTML('<div style="margin-top:16px"></div>')

    # Anamnese
    gr.HTML('<div class="result-header-anamnese">Anamnese</div>')
    out_dialogue = gr.Markdown(value="", elem_classes=["result-body"])

    # Raciocinio Clinico
    gr.HTML('<div class="result-header-clinical">Raciocinio Clinico</div>')
    out_clinical = gr.Markdown(value="", elem_classes=["result-body"])

    # Seguranca Medicamentosa
    gr.HTML('<div class="result-header-safety">Seguranca Medicamentosa</div>')
    out_medication = gr.Markdown(value="", elem_classes=["result-body"])

    # -- Export ----------------------------------------------------------------
    export_btn = gr.Button("Exportar SOAP", variant="secondary", size="sm")
    soap_out = gr.Markdown(visible=False)

    # -- API Key (bottom, out of the way) --------------------------------------
    with gr.Accordion("Configuracoes", open=not bool(GOOGLE_API_KEY_ENV)):
        api_key_input = gr.Textbox(
            label="Google API Key",
            placeholder="Obtenha em aistudio.google.com",
            type="password",
            value=GOOGLE_API_KEY_ENV,
        )
        model_selector = gr.Dropdown(
            choices=[
                "gemma-4-31b-it",
                "gemma-4-26b-a4b-it",
                "gemma-3-27b-it",
            ],
            value="gemma-4-31b-it",
            label="Modelo",
        )

    gr.HTML("""
    <div class="disclaimer">
        Ferramenta de suporte clinico -- nao substitui o julgamento medico.
        Toda conduta deve ser validada por profissional habilitado.<br/>
        Inspirado no AMIE (Google DeepMind) -- Powered by Gemma 4 via Google AI.
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
        outputs=[out_dialogue, out_clinical, out_medication],
    )

    example_btn.click(
        fn=load_example,
        outputs=[symptoms, history, medications],
    )

    export_btn.click(
        fn=export_soap,
        inputs=[symptoms, history, medications, out_dialogue, out_clinical, out_medication],
        outputs=[soap_out],
    )

if __name__ == "__main__":
    demo.launch()
