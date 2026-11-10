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

ai_client = OpenAI(api_key=GROQ_KEY.strip(), base_url="https://api.groq.com/openai/v1")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = """Eres VESTA, la Asesora Inmobiliaria Virtual de AINovaX.
Tu trabajo es calificar clientes, responder sobre propiedades, captar inmuebles y agendar visitas.
- Habla en español amable, empático, profesional y enfocado en la venta o visita.
- Si el cliente menciona su ciudad (como Armenia, Bogotá, etc.) o presupuesto, pregúntale qué tipo de inmueble busca (casa, apartamento, lote) y cuántas habitaciones.
- Si quiere agendar visita, pide Nombre, Teléfono y fecha/hora preferida."""

class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        models_data = ai_client.models.list()
        chat_models = [m.id for m in models_data.data if any(k in m.id.lower() for k in ["llama", "mixtral", "gemma"]) and "guard" not in m.id.lower() and "orpheus" not in m.id.lower()]
        
        try:
            inmuebles = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni").limit(5).execute()
            catalogo = f"\nPropiedades en catálogo:\n{json.dumps(inmuebles.data, ensure_ascii=False)}" if inmuebles.data else ""
        except Exception:
            catalogo = ""

        messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}{catalogo}"}]
        for msg in req.chat_history:
            messages.append(msg)
        messages.append({"role": "user", "content": req.message})

        ultimo_error = ""
        for mod in chat_models:
            try:
                response = ai_client.chat.completions.create(model=mod, messages=messages, temperature=0.7)
                return {"reply": response.choices[0].message.content}
            except Exception as err:
                ultimo_error = str(err)
                continue

        return {"reply": f"Detalle modelos: {ultimo_error}"}
    except Exception as e:
        return {"reply": f"Aviso de conexión: {str(e)}"}
