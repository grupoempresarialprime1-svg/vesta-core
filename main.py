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

# Permitir conexiones desde cualquier web o dominio (HostGator, WhatsApp, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar clientes
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"), 
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)

# -------------------------------------------------------------
# 1. HERRAMIENTAS (TOOLS DE VESTA)
# -------------------------------------------------------------
def tool_buscar_inmuebles(query_testo: str, citta: str = None, prezzo_max: float = None):
    """Busca inmuebles usando búsqueda semántica con vectores."""
    emb_resp = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=query_testo
    )
    vector = emb_resp.data[0].embedding
    
    rpc_params = {
        "query_embedding": vector,
        "filtro_citta": citta,
        "filtro_prezzo_max": prezzo_max,
        "limite": 3
    }
    res = supabase.rpc("buscar_inmuebles_vectorial", rpc_params).execute()
    return res.data if res.data else []

def tool_analizar_foto_fachada(image_url: str):
    """Analiza la fachada del inmueble con visión artificial."""
    prompt = (
        "Analizza l'immagine della facciata. Restituisci SOLO un JSON valido con le seguenti chiavi: "
        "tipo_immobile, stile_architettonico, numero_piani_stimato, stato_conservazione."
    )
    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            }
        ],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)

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
            "description": "Cerca immobili nel database usando ricerca semantica e filtri.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_testo": {"type": "string", "description": "Descrizione o requisiti dell'immobile"},
                    "citta": {"type": "string", "description": "Città dove cercare l'immobile"},
                    "prezzo_max": {"type": "number", "description": "Budget massimo"}
                },
                "required": ["query_testo"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analizar_foto_fachada",
            "description": "Analizza una foto di una facciata inviata dall'utente.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL pubblico dell'immagine"}
                },
                "required": ["image_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "agendar_visita",
            "description": "Fissa un appuntamento per visitare un immobile specifico.",
            "parameters": {
                "type": "object",
                "properties": {
                    "immobile_id": {"type": "integer", "description": "ID dell'immobile"},
                    "nome_cliente": {"type": "string", "description": "Nome completo del cliente"},
                    "telefono": {"type": "string", "description": "Numero di telefono"},
                    "data_ora": {"type": "string", "description": "Data e ora ISO (es. 2026-11-15T10:00:00)"}
                },
                "required": ["immobile_id", "nome_cliente", "telefono", "data_ora"]
            }
        }
    }
]

SYSTEM_PROMPT = """
Sei VESTA, l'Agente Virtuale Autonomo specializzato nel settore immobiliare. 
Il tuo obiettivo è ESEGUIRE processi completi senza supervisione: qualificare clienti, cercare immobili, captare proprietà da foto e fissare appuntamenti.
- Lingua: Rispondi sempre nella lingua parlata dall'utente (Spagnolo neutro per LATAM o Spagna, Italiano se l'utente parla italiano).
- Tono: Professionale, commerciale, empatico, focalizzato sulla chiusura dell'affare (conversione).
- LIVELLO 1: Rispondi basandoti SOLO sui dati restituiti dal tool di ricerca. Non inventare proprietà inesistenti.
- MANCANZA DATI CRITICI: Se l'utente vuole fissare un appuntamento ma manca data/ora/telefono, NON chiamare agendar_visita, chiedi prima con gentilezza i dati mancanti.
- TRASPARENZA: Non dire mai "Sto usando uno strumento", parla come un consulente immobiliare umano ed esperto.
"""

# -------------------------------------------------------------
# 2. ENDPOINTS
# -------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    image_url: str = None
    chat_history: list = []

@app.get("/")
def home():
    return {"status": "online", "agent": "VESTA", "version": "1.0"}

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.chat_history)
    
    if req.image_url:
        messages.append({
            "role": "user",
            "content": [
                {"type": "text", "text": req.message or "Ho caricato questa foto della facciata."},
                {"type": "image_url", "image_url": {"url": req.image_url}}
            ]
        })
    else:
        messages.append({"role": "user", "content": req.message})

    response = openai_client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto"
    )
    
    response_message = response.choices[0].message
    
    if response_message.tool_calls:
        messages.append(response_message)
        
        for tool_call in response_message.tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
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
            
        second_response = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )
        return {"reply": second_response.choices[0].message.content}
        
    return {"reply": response_message.content}
