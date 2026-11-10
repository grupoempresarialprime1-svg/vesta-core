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

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

SYSTEM_PROMPT = """
Sei VESTA, l'Agente Virtuale Autonomo di AINovaX specializzato nel settore immobiliare. 
Il tuo obiettivo è ESEGUIRE processi completi: qualificare clienti, cercare immobili, captare proprietà e fissare appuntamenti.
- Lingua: Rispondi sempre nella lingua parlata dall'utente (Spagnolo neutro per LATAM o Spagna, Italiano se l'utente parla italiano).
- Tono: Professionale, commerciale, empatico, amichevole e focalizzato sulla chiusura dell'affare (conversione).
- Non inventare proprietà inesistenti. Se l'utente chiede case, usa quelle fornite nel contesto.
- Se l'utente vuole fissare un appuntamento o visita, chiedi con gentilezza: Nome, Telefono e la data/ora preferita.
"""

class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0", "engine": "Google Gemini Direct"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    # 1. Obtener inmuebles disponibles en Supabase para darle memoria real a Vesta
    try:
        inmuebles_res = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni").limit(5).execute()
        inmuebles_data = inmuebles_res.data if inmuebles_res.data else []
    except Exception:
        inmuebles_data = []

    contexto_inmuebles = f"\n\nIMMOBILI ATTUALMENTE NEL DATABASE:\n{json.dumps(inmuebles_data, ensure_ascii=False)}" if inmuebles_data else "\n\n(Nessun immobile ancora caricato nel database. Se chiedono case, dì che stiamo aggiornando il catalogo e chiedi che zona cercano)."

    prompt_completo = f"{SYSTEM_PROMPT}{contexto_inmuebles}\n\nCliente dice: {req.message}"

    # 2. Llamada directa a Google Gemini (Prueba gemini-2.0-flash y luego gemini-1.5-flash)
    modelos = ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        for modelo in modelos:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{modelo}:generateContent?key={GEMINI_API_KEY}"
            payload = {
                "contents": [
                    {
                        "parts": [{"text": prompt_completo}]
                    }
                ]
            }
            try:
                resp = await client.post(url, json=payload)
                if resp.status_code == 200:
                    data = resp.json()
                    respuesta_texto = data["candidates"][0]["content"]["parts"][0]["text"]
                    return {"reply": respuesta_texto}
            except Exception:
                continue

    return {"reply": "¡Hola! Soy Vesta, tu asesora inmobiliaria. ¿En qué ciudad o zona estás buscando tu próxima propiedad hoy?"}
