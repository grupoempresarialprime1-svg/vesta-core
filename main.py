import os
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="VESTA Core Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

# Conectar al motor ultra rápido y gratis de Groq
ai_client = OpenAI(
    api_key=GROQ_KEY.strip(),
    base_url="https://api.groq.com/openai/v1"
)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = """
Eres VESTA, la Asesora Inmobiliaria Virtual de AINovaX. 
Tu trabajo es calificar clientes, responder sobre propiedades, captar inmuebles y agendar visitas.
- Habla siempre en español amable, empático, profesional y enfocado en vender.
- Si el cliente te dice de qué ciudad es (como Armenia, Medellín, etc.) o si quiere "comprar" o "alquilar", pregúntale qué tipo de inmueble busca (casa, apartamento, lote), cuántas habitaciones necesita y cuál es su presupuesto.
- Si quiere agendar una cita o visita, pídele su Nombre completo, Teléfono y fecha/hora preferida.
"""

class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0", "engine": "Groq LLaMA 3.3 Free"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        # Consultar si hay casas en Supabase
        inmuebles = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni").limit(5).execute()
        catalogo = f"\nPropiedades en catálogo:\n{json.dumps(inmuebles.data, ensure_ascii=False)}" if inmuebles.data else "\n(Catálogo actualizándose)."
    except Exception:
        catalogo = ""

    prompt_sistema = f"{SYSTEM_PROMPT}{catalogo}"

    messages = [{"role": "system", "content": prompt_sistema}]
    for msg in req.chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": req.message})

    try:
        response = ai_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.7
        )
        return {"reply": response.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Aviso de conexión: {str(e)}"}
