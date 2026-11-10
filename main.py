import os
import json
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
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

raw_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
GEMINI_API_KEY = raw_key.strip()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = """
Eres VESTA, la Asesora Inmobiliaria Virtual de AINovaX. 
Tu objetivo es calificar clientes, responder sobre propiedades, captar inmuebles y agendar visitas.
- Idioma: Habla siempre en español amable, profesional, empático y comercial.
- Si el cliente te dice de qué ciudad es (como Armenia, Medellín, Bogotá, etc.), salúdalo con entusiasmo por esa ciudad y pregúntale qué tipo de inmueble busca (casa, apartamento, lote) o su presupuesto.
- Si quiere agendar una cita o visita, pídele su Nombre completo, Teléfono y qué día y hora le queda mejor.
"""

class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # Obtener inmuebles de Supabase si existen
    try:
        inmuebles_res = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni").limit(5).execute()
        inmuebles_data = inmuebles_res.data if inmuebles_res.data else []
    except Exception:
        inmuebles_data = []

    info_inmuebles = f"\nInmuebles en catálogo:\n{json.dumps(inmuebles_data, ensure_ascii=False)}" if inmuebles_data else "\n(Catálogo actualizándose)."
    prompt_completo = f"{SYSTEM_PROMPT}\n{info_inmuebles}\n\nCliente: {req.message}\nVESTA:"

    # Llamar a Google Gemini
    modelos = ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro"]
    
    async with httpx.AsyncClient(timeout=25.0) as client:
        for modelo in modelos:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [{"parts": [{"text": prompt_completo}]}]
            }
            try:
                resp = await client.post(url, json=payload)
                print(f"GOOGLE DEBUG ({modelo}): Status {resp.status_code}")
                if resp.status_code == 200:
                    data = resp.json()
                    texto = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"reply": texto}
                else:
                    print(f"GOOGLE ERROR BODY: {resp.text}")
            except Exception as e:
                print(f"ERROR EXCEPTION: {e}")
                continue

    # Respuesta comercial inteligente si la clave de Google no responde
    msg_lower = req.message.lower()
    if "armenia" in msg_lower or "bogota" in msg_lower or "medellin" in msg_lower or "cali" in msg_lower:
        return {"reply": f"¡Excelente! Qué gran zona. En este momento tenemos varias opciones disponibles para ti. ¿Buscas casa o apartamento, y cuál es tu presupuesto aproximado?"}
    elif "visita" in msg_lower or "cita" in msg_lower or "ver" in msg_lower:
        return {"reply": "¡Con mucho gusto coordinamos la visita! Por favor compárteme tu Nombre completo, tu número de Teléfono y qué día te gustaría visitarlo."}
    else:
        return {"reply": f"¡Hola! Un gusto saludarte. Soy VESTA, tu asesora inmobiliaria. ¿Estás buscando comprar, alquilar o vender una propiedad hoy?"}
