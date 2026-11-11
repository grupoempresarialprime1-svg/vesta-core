import os
import json
import re
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client, Client

load_dotenv()

app = FastAPI(title="Inmobiliaria La Santa María - Core Engine")

# Permitir conexiones desde cualquier dominio (HostGator, local, etc.)
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

SYSTEM_PROMPT = """Eres VESTA, la Asesora Inmobiliaria Virtual de Inmobiliaria La Santa María en Armenia, Quindío.
Tu objetivo es atender clientes, responder sobre propiedades de nuestro catálogo, captar propietarios y agendar visitas sin cruzar horarios.
- Tono: Muy empático, profesional, respetuoso, comercial y enfocado en la venta o arrendamiento seguro.
- Si el cliente pregunta por inmuebles, utiliza prioritariamente las propiedades de nuestro catálogo real.
- Si el cliente quiere agendar una visita: pídele su Nombre completo, Teléfono, qué propiedad desea ver y el día/hora que prefiere.
- Si el cliente es un propietario que quiere poner su inmueble en arriendo o venta, explícale las ventajas de Inmobiliaria La Santa María: póliza garantizada, pago puntual y cero estrés con inquilinos."""

# Helper para ejecutar llamadas a la IA con rotación de modelos de texto
def consultar_ia(messages: list, temperature: float = 0.7) -> str:
    try:
        models_data = ai_client.models.list()
        chat_models = [m.id for m in models_data.data if any(k in m.id.lower() for k in ["llama", "mixtral", "gemma"]) and "guard" not in m.id.lower() and "orpheus" not in m.id.lower() and "vision" not in m.id.lower()]
    except Exception:
        chat_models = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "llama3-8b-8192"]

    for mod in chat_models:
        try:
            res = ai_client.chat.completions.create(model=mod, messages=messages, temperature=temperature)
            return res.choices[0].message.content
        except Exception:
            continue
    return "¡Hola! Estoy atenta a tu solicitud. ¿En qué zona de Armenia buscas o qué tipo de inmueble te interesa?"

# Helper para ejecutar llamadas a la IA con Visión Artificial (Llama 3.2 Vision)
def consultar_vision_ia(image_base64: str, prompt_text: str) -> str:
    vision_models = ["llama-3.2-11b-vision-preview", "llama-3.2-90b-vision-preview"]
    
    if not image_base64.startswith("data:image"):
        image_url = f"data:image/jpeg;base64,{image_base64}"
    else:
        image_url = image_base64

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]
        }
    ]

    for model_name in vision_models:
        try:
            res = ai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.1,
                max_tokens=600
            )
            return res.choices[0].message.content
        except Exception as e:
            print(f"Vision model {model_name} error: {e}")
            continue
    return ""

# ---------------------------------------------------------------------
# 1. ENDPOINT PRINCIPAL: CHAT DE VESTA (INBOUND LEADS)
# ---------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str
    phone: str = None
    chat_history: list = []

@app.post("/chat")
async def chat_endpoint(req: ChatRequest):
    try:
        inmuebles = supabase.table("immobili").select("id, titolo, descrizione, prezzo, tipo_operacion, barrio, camere, bagni").eq("estado", "Disponible").limit(6).execute()
        catalogo = f"\n\nCATÁLOGO ACTUAL EN LA BASE DE DATOS:\n{json.dumps(inmuebles.data, ensure_ascii=False)}" if inmuebles.data else ""
    except Exception:
        catalogo = ""

    messages = [{"role": "system", "content": f"{SYSTEM_PROMPT}{catalogo}"}]
    for msg in req.chat_history:
        messages.append(msg)
    messages.append({"role": "user", "content": req.message})

    respuesta = consultar_ia(messages)

    msg_lower = req.message.lower()
    if any(k in msg_lower for k in ["mi numero", "mi telefono", "me llamo", "mi nombre", "300", "310", "311", "312", "313", "314", "315", "316", "317", "318", "319", "320"]):
        try:
            supabase.table("leads_crm").insert({
                "nombre": "Cliente Web / Chat",
                "telefono": req.phone or "Por confirmar",
                "interes": "Arriendo" if "arriendo" in msg_lower else "Compra",
                "etapa_embudo": "Calificado",
                "ultimo_mensaje_cliente": req.message,
                "recomendacion_ia": "Llamar para agendar visita."
            }).execute()
        except Exception:
            pass

    return {"reply": respuesta}

# ---------------------------------------------------------------------
# 2. CAPTADOR DE CARTELES CON VISIÓN ARTIFICIAL (SEPARA PRECIOS DE TELÉFONOS)
# ---------------------------------------------------------------------
class CartelRequest(BaseModel):
    image_base64: str = None
    foto_url: str = None
    texto_cartel_manual: str = None
    ciudad: str = "Armenia"
    barrio: str = "Por identificar"

@app.post("/captar-cartel")
async def captar_cartel(req: CartelRequest):
    resp_texto = ""
    
    # Si viene foto, ejecutar el modelo multimodal de visión artificial
    if req.image_base64:
        prompt_vision = """Eres un sistema de visión artificial de alta precisión para bienes raíces en Colombia.
Analiza la fotografía de este letrero o aviso inmobiliario de la calle.

REGLAS ESTRICTAS PARA NÚMEROS:
En la imagen pueden aparecer muchos números diferentes:
- Precios (ej: $150.000.000, 1.200.000, 450 millones) -> ESTO ES UN PRECIO, NUNCA ES EL TELÉFONO.
- Áreas o metrajes (ej: 80m2, 120 mts, 240 m²) -> ES UN ÁREA, NUNCA ES EL TELÉFONO.
- Cantidad de alcobas o baños (ej: 3 hab, 2 baños, 4 alcobas) -> NUNCA ES EL TELÉFONO.
- Direcciones o nomenclatura (ej: Calle 14 # 23-45, Cra 19, Piso 3) -> NUNCA ES EL TELÉFONO.

TU MISIÓN:
1. Extraer el NÚMERO DE TELÉFONO CELULAR COLOMBIANO de contacto del dueño o anunciante (número de 10 dígitos que comienza por 3, ej: 310..., 315..., 320..., 300... o fijo de 7 dígitos).
2. Identificar si es 'Arriendo' o 'Venta'.
3. Identificar el tipo de inmueble (Casa, Apartamento, Local, Lote).
4. Identificar si menciona algún barrio o sector de Armenia.

Responde ÚNICAMENTE un JSON válido con esta estructura:
{
  "telefono": "3101234567",
  "tipo_operacion": "Arriendo",
  "tipo_inmueble": "Apartamento",
  "barrio": "Laureles"
}"""
        resp_texto = consultar_vision_ia(req.image_base64, prompt_vision)

    # Fallback si no hay imagen o si falló la visión
    if not resp_texto and (req.texto_cartel_manual or req.foto_url):
        prompt_texto = f"""Analiza la siguiente información de un letrero de 'Se Vende' o 'Se Arrienda':
        Información: {req.texto_cartel_manual or req.foto_url}
        Distingue teléfonos de precios o metrajes. Extrae teléfono celular colombiano (10 dígitos con 3), tipo_operacion ('Arriendo' o 'Venta'), tipo_inmueble y barrio.
        Responde ÚNICAMENTE un JSON:
        {{"telefono": "3101234567", "tipo_operacion": "Arriendo", "tipo_inmueble": "Casa", "barrio": "Armenia"}}"""
        resp_texto = consultar_ia([{"role": "user", "content": prompt_texto}], temperature=0.2)

    data = {}
    if resp_texto:
        try:
            clean_json = resp_texto[resp_texto.find("{"):resp_texto.rfind("}")+1]
            data = json.loads(clean_json)
        except Exception:
            pass

    tel_raw = str(data.get("telefono") or req.texto_cartel_manual or "")
    match_tel = re.search(r'3\d{9}', re.sub(r'\D', '', tel_raw))
    tel_clean = match_tel.group(0) if match_tel else re.sub(r'\D', '', tel_raw)
    
    tipo_op = data.get("tipo_operacion") or ("Arriendo" if "arriend" in (req.texto_cartel_manual or "").lower() else "Venta")
    barrio_detectado = data.get("barrio") or req.barrio or "Armenia"
    
    if tipo_op == "Arriendo":
        msg = f"Hola, buenas tardes. Vimos que tienes tu propiedad en {barrio_detectado} en arriendo. Somos Inmobiliaria La Santa María en Armenia. Te ofrecemos encargarnos de todo con póliza de arrendamiento garantizada, pago puntual cada mes y cero estrés lidiando con inquilinos. ¿Te gustaría que conversemos unos minutos sin compromiso?"
    else:
        msg = f"Hola, buenas tardes. Vimos que tienes tu propiedad en {barrio_detectado} en venta. Somos Inmobiliaria La Santa María en Armenia. Contamos con clientes calificados listos para comprar y nos encargamos de todo el trámite legal, estudio de títulos y visitas. ¿Te gustaría que conversemos unos minutos sin compromiso?"

    wa_tel = ("57" + tel_clean) if len(tel_clean) == 10 else tel_clean
    whatsapp_url = f"https://wa.me/{wa_tel}?text={json.dumps(msg)[1:-1]}" if tel_clean else ""

    try:
        supabase.table("prospectos_captacion").insert({
            "foto_cartel_url": req.foto_url or ("Imagen analizada con Visión Artificial" if req.image_base64 else None),
            "telefono_detectado": tel_clean,
            "tipo_negocio": tipo_op,
            "ciudad": req.ciudad,
            "direccion_aproximada": barrio_detectado,
            "estado_contacto": "Pendiente",
            "mensaje_enviado": msg
        }).execute()
    except Exception:
        pass

    return {
        "status": "success",
        "telefono": tel_clean,
        "tipo_operacion": tipo_op,
        "barrio": barrio_detectado,
        "tipo_inmueble": data.get("tipo_inmueble", "Inmueble"),
        "mensaje_whatsapp": msg,
        "whatsapp_link": whatsapp_url
    }

# ---------------------------------------------------------------------
# 3. AGENDAR VISITA (SIN COLISIÓN DE HORARIOS)
# ---------------------------------------------------------------------
class VisitaRequest(BaseModel):
    immobile_id: int
    nome_cliente: str
    telefono: str
    data_ora: str

@app.post("/agendar-visita")
async def agendar_visita(req: VisitaRequest):
    try:
        visitas_existentes = supabase.table("appuntamenti").select("*").eq("immobile_id", req.immobile_id).execute()
        for v in visitas_existentes.data:
            if v.get("data_ora") == req.data_ora:
                return {
                    "status": "colision",
                    "mensaje": "Ese horario ya se encuentra reservado para esta propiedad. Por favor elige otro horario disponible."
                }
    except Exception:
        pass

    try:
        res = supabase.table("appuntamenti").insert({
            "immobile_id": req.immobile_id,
            "nome_cliente": req.nome_cliente,
            "telefono_cliente": req.telefono,
            "data_ora": req.data_ora,
            "estado": "Confirmada"
        }).execute()

        supabase.table("leads_crm").insert({
            "nombre": req.nome_cliente,
            "telefono": req.telefono,
            "etapa_embudo": "Visita Agendada",
            "ultimo_mensaje_cliente": f"Visita agendada para el inmueble #{req.immobile_id}"
        }).execute()

        return {"status": "success", "visita": res.data[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ---------------------------------------------------------------------
# 4. CRM DASHBOARD & COPILOTO DE IA
# ---------------------------------------------------------------------
@app.get("/crm/dashboard")
async def crm_dashboard():
    try:
        leads = supabase.table("leads_crm").select("*").order("ultimo_contacto", desc=True).limit(50).execute().data
        inmuebles = supabase.table("immobili").select("*").order("id", desc=True).limit(50).execute().data
        visitas = supabase.table("appuntamenti").select("*, immobili(titolo, citta)").order("data_ora", desc=False).limit(20).execute().data
        captaciones = supabase.table("prospectos_captacion").select("*").order("creado_el", desc=True).limit(20).execute().data

        embudo = {
            "Nuevos": [l for l in leads if l.get("etapa_embudo") in ["Nuevo", None]],
            "Calificados": [l for l in leads if l.get("etapa_embudo") == "Calificado"],
            "Visita Agendada": [l for l in leads if l.get("etapa_embudo") == "Visita Agendada"],
            "Frios": [l for l in leads if l.get("etapa_embudo") == "Frio"],
            "Cerrados": [l for l in leads if l.get("etapa_embudo") == "Cerrado"]
        }

        return {
            "embudo": embudo,
            "inmuebles": inmuebles,
            "visitas": visitas,
            "captaciones": captaciones,
            "resumen": {
                "total_inmuebles": len(inmuebles),
                "total_leads": len(leads),
                "visitas_pendientes": len(visitas),
                "captaciones_pendientes": len([c for c in captaciones if c.get("estado_contacto") == "Pendiente"])
            }
        }
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------------------
# 5. BOTÓN MÁGICO: REACTIVAR CLIENTE FRÍO CON IA
# ---------------------------------------------------------------------
class ReactivarRequest(BaseModel):
    lead_id: int

@app.post("/crm/reactivar-lead")
async def reactivar_lead(req: ReactivarRequest):
    lead = supabase.table("leads_crm").select("*").eq("id", req.lead_id).execute().data[0]
    inmuebles = supabase.table("immobili").select("titolo, prezzo, barrio, tipo_operacion").limit(3).execute().data

    prompt = f"""Eres VESTA de Inmobiliaria La Santa María. Redacta un mensaje de WhatsApp corto, cercano y tentador para retomar a un cliente que se enfrió.
    Cliente: {lead.get('nombre')}, busca: {lead.get('interes')} en {lead.get('ciudad_interes')}.
    Nuevas opciones disponibles para tentarlo: {json.dumps(inmuebles, ensure_ascii=False)}.
    Genera SOLO el texto para enviar por WhatsApp."""

    mensaje = consultar_ia([{"role": "user", "content": prompt}], temperature=0.7)
    
    telefono_clean = "".join([c for c in str(lead.get("telefono", "")) if c.isdigit()])
    if len(telefono_clean) == 10 and not telefono_clean.startswith("57"):
        telefono_clean = "57" + telefono_clean

    whatsapp_link = f"https://wa.me/{telefono_clean}?text={json.dumps(mensaje)[1:-1]}"

    return {
        "mensaje": mensaje,
        "whatsapp_link": whatsapp_link
    }

# ---------------------------------------------------------------------
# 6. CATÁLOGO PÚBLICO PARA LA WEB Y EL MAPA
# ---------------------------------------------------------------------
@app.get("/propiedades")
async def listar_propiedades(tipo: str = None, operacion: str = None, ciudad: str = "Armenia"):
    try:
        q = supabase.table("immobili").select("*").eq("estado", "Disponible")
        if operacion:
            q = q.eq("tipo_operacion", operacion)
        if tipo:
            q = q.eq("tipo_inmueble", tipo)
        if ciudad:
            q = q.ilike("citta", f"%{ciudad}%")
        res = q.execute()
        return res.data
    except Exception as e:
        return []

@app.get("/")
def root():
    return {"status": "online", "agency": "Inmobiliaria La Santa María", "agent": "VESTA", "version": "3.0 Vision AI"}
