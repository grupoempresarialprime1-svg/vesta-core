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

# Inicializar Google Gemini (100% GRATIS)
gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
ai_client = OpenAI(
    api_key=gemini_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# -------------------------------------------------------------
# 1. HERRAMIENTAS (TOOLS DE VESTA)
# -------------------------------------------------------------
def tool_buscar_inmuebles(query_testo: str = "", citta: str = None, prezzo_max: float = None):
    """Busca inmuebles en Supabase por ciudad, precio o características."""
    try:
        q = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni")
        if citta:
            q = q.ilike("citta", f"%{citta}%")
        if prezzo_max:
            q = q.lte("prezzo", prezzo_max)
        res = q.limit(3).execute()
        
        if res.data and len(res.data) > 0:
            return res.data
            
        fallback = supabase.table("immobili").select("id, titolo, descrizione, prezzo, citta, camere, bagni").limit(3).execute()
        return fallback.data if fallback.data else []
    except Exception as e:
        return {"error": str(e)}

def tool_analizar_foto_fachada(image_url: str):
    """Analiza la fachada del inmueble con visión artificial de Gemini."""
    prompt = (
        "Analizza l'immagine della facciata. Restituisci SOLO un JSON valido con le seguenti chiavi: "
        "tipo_immobile, stile_architettonico, numero_piani_stimato, stato_conservazione."
    )
    response = ai_client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ]
    )
    return response.choices[0].message.content

def tool_agendar_visita(immobile_id: int, nome_cliente: str, telefono: str, data_ora: str):
    """Guarda la cita en la tabla appuntamenti de Supabase."""
    data = {
        "immobile_id": immobile_id,
        "nome_cliente": nome_cliente,
        "telefono_cliente": telefono,
        "data_ora": data_ora,
        "stato_confermato": True
    }
    res = supabase.table("appuntamenti").insert(data).execute()
    return {"status": "success", "reserva": res.data[0] if res.data else {}}

AVAILABLE_TOOLS = {
    "buscar_inmuebles": tool_buscar_inmuebles,
    "analizar_foto_fachada": tool_analizar_foto_fachada,
    "agendar_visita": tool_agendar_visita
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "buscar_inmuebles",
            "description": "Cerca immobili nel database usando filtri di città, budget o descrizione.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_testo": {"type": "string", "description": "Descrizione dell'immobile"},
                    "citta": {"type": "string", "description": "Città dove cercare"},
                    "prezzo_max": {"type": "number", "description": "Budget massimo"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analizar_foto_fachada",
            "description": "Analizza una foto di facciata per estrarre dettagli.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL dell'immagine"}
                },
                "required": ["image_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agendar_visita",
            "description": "Fissa un appuntamento per visitare un immobile.",
            "parameters": {
                "type": "object",
                "properties": {
                    "immobile_id": {"type": "integer", "description": "ID immobile"},
                    "nome_cliente": {"type": "string", "description": "Nome cliente"},
                    "telefono": {"type": "string", "description": "Telefono cliente"},
                    "data_ora": {"type": "string", "description": "Data e ora"}
                },
                "required": ["immobile_id", "nome_cliente", "telefono", "data_ora"]
            }
        }
    }
]

SYSTEM_PROMPT = """
Sei VESTA, l'Agente Virtuale Autonomo specializzato nel settore immobiliare. 
Il tuo obiettivo è ESEGUIRE processi completi: qualificare clienti, cercare immobili, captare proprietà e fissare appuntamenti.
- Lingua: Rispondi sempre nella lingua parlata dall'utente (Spagnolo neutro per LATAM o Spagna, Italiano se l'utente parla italiano).
- Tono: Professionale, commerciale, empatico, amichevole e focalizzato sulla chiusura dell'affare.
- LIVELLO 1: Rispondi basandoti SOLO sui dati del database. Non inventare proprietà inesistenti.
- MANCANZA DATI CRITICI: Se l'utente vuole fissare un appuntamento ma manca data/ora/telefono, NON chiamare agendar_visita, chiedi prima con gentilezza i dati mancanti.
"""

class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0", "engine": "Google Gemini Free"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.chat_history)
    
    if req.image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": req.message or "Analizza questa facciata."},
                {"type": "image_url", "image_url": {"url": req.image_url}}
            ]
        })
    else:
        messages.append({"role": "user", "content": req.message})

    response = ai_client.chat.completions.create(
        model="gemini-1.5-flash",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    if response_message.tool_calls:
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
            
            tool_func = AVAILABLE_TOOLS.get(function_name)
            if tool_func:
                try:
                    tool_output = tool_func(**function_args)
                except Exception as e:
                    tool_output = {"error": str(e)}
            else:
                tool_output = {"error": "Tool non trovato"}
                
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(tool_output)
            })
            
        second_response = ai_client.chat.completions.create(
            model="gemini-1.5-flash",
            messages=messages
        )
        return {"reply": second_response.choices[0].message.content}
        
    return {"reply": response_message.content}
