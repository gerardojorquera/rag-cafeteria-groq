import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
# 1. CORRECCIÓN: Importación oficial y moderna desde langchain_huggingface
from langchain_huggingface import HuggingFaceEndpointEmbeddings 
from langchain_community.vectorstores import FAISS 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
import gradio as gr 

load_dotenv()
# --- LÍNEAS DE PRUEBA TEMPORALES ---
print("=== VERIFICANDO VARIABLES DE ENTORNO ===")
print(f"¿Groq Key cargada?: {os.getenv('GROQ_API_KEY') is not None}")
print(f"¿HF Token cargado?: {os.getenv('HF_TOKEN') is not None}")
if os.getenv('HF_TOKEN'):
    print(f"Inicio del HF Token: {os.getenv('HF_TOKEN')[:7]}...")
print("=========================================")

# Cargar la API Key de Groq desde tu archivo .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Inicialización de Llama 3 en la nube de Groq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=GROQ_API_KEY,
    temperature=0.3
)

# 2. CORRECCIÓN: Conector Serverless oficial (Consume 0 MB de RAM en Render)
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN")
)

# Carga del documento de la cafetería
with open("cafeteria.txt", "r", encoding="utf-8") as f:
    documento = f.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(documento)
vectorstore = FAISS.from_texts(chunks, embeddings) 
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k":4})

prompt = ChatPromptTemplate.from_template("""Eres el asistente virtual de Cafe Aurora. Tu trabajo es responder preguntas
de los clientes UNICAMENTE usando la informacion proporcionada en el contexto.

Reglas estrictas:
1. SOLO responde con informacion que esté en el contexto.
2. Si la pregunta no se puede responder con el contexto, di:
"Lo siento, no tengo esa informacion. Te recomiendo contactarnos
por WhatsApp al +56 9 8765 4321 o por email a contacto@cafeaurora.cl"
3. Sé amable, conciso y útil.
4. Si preguntan precios, siempre menciona el precio exacto.
5. Responde en español.

Contexto:
{context}

Pregunta del cliente: {question}

Respuesta:""")

def format_docs(docs): 
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def responder(mensaje, historia):
    # Procesamos la pregunta en la cadena RAG con tu LLM (Groq)
    respuesta_ia = rag_chain.invoke(mensaje)
    
    # Estructura obligatoria usando objetos gr.ChatMessage
    # Esto elimina el error de "Data incompatible with messages format" de raíz
    historia.append(gr.ChatMessage(role="user", content=mensaje))
    historia.append(gr.ChatMessage(role="assistant", content=respuesta_ia))
    
    return "", historia  # Limpia la caja de texto y actualiza el chat

# Creación de la interfaz compacta con gr.Blocks
with gr.Blocks(title="Asistente Virtual - Café Aurora") as demo:
    gr.Markdown("# Asistente Virtual - Café Aurora")
    gr.Markdown("Pregúntame sobre nuestro menú, horarios, ubicación, eventos y más.")
    
    # Contenedor del historial del chat estándar
    chatbot = gr.Chatbot(height=360)
    
    # Campo de entrada de texto para el usuario
    txt_input = gr.Textbox(
        show_label=False, 
        placeholder="Escribe tu pregunta aquí y presiona Enter...",
        container=False
    )
    
    gr.Markdown("### 💡 Preguntas Frecuentes:")

        # 1. Los ejemplos en gr.Examples requieren estar en formato de lista de listas [[str]]
    ejemplos_lista = [
        ["¿Horario los sábados?"],
        ["¿Opciones veganas?"],
        ["¿Precio cappuccino?"],
        ["¿Hacen delivery?"],
        ["¿Tienen wifi?"],
    ]
    
    # 2. Creamos el componente nativo de Gradio
    ejemplos_componente = gr.Examples(
        examples=ejemplos_lista,
        inputs=txt_input,  # Vincula el clic del ejemplo directamente a la caja de texto
        label=None
    )
    
    # 3. TRUCO DE CONEXIÓN: Hacemos que al hacer clic en un ejemplo, 
    # se ejecute AUTOMÁTICAMENTE la función 'responder' y luego se devuelva el foco
    ejemplos_componente.dataset.click(
        fn=responder,
        inputs=[txt_input, chatbot],
        outputs=[txt_input, chatbot]
    ).then(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { document.querySelector('textarea, input[type=text]').focus(); }"
    )

    """    
    ejemplos = [
        "¿Horario los sábados?",
        "¿Opciones veganas?",
        "¿Precio cappuccino?",
        "¿Hacen delivery?",
        "¿Tienen wifi?",
    ]

    # Fila compacta para visualizar todo en una sola pantalla sin scroll vertical
    with gr.Row(variant="compact"):
        for texto_ejemplo in ejemplos:
            btn = gr.Button(texto_ejemplo, variant="secondary", size="sm", min_width=10)
            
            # Flujo: Procesa la IA con los mensajes estructurados y luego hace foco con JS
            btn.click(
                fn=responder, 
                inputs=[btn, chatbot], 
                outputs=[txt_input, chatbot]
            ).then(
                fn=None,
                inputs=None,
                outputs=None,
                js="() => { document.querySelector('textarea, input[type=text]').focus(); }"
            )
    """

    # Acción al presionar 'Enter' en la caja de texto principal
    txt_input.submit(
        fn=responder, 
        inputs=[txt_input, chatbot], 
        outputs=[txt_input, chatbot]
    )

    # ==========================================
    # NUEVO CAMBIO: Foco automático al abrir la página
    # ==========================================
    demo.load(
        fn=None,
        inputs=None,
        outputs=None,
        js="() => { setTimeout(() => { document.querySelector('textarea, input[type=text]').focus(); }, 500); }"
    )    

if __name__ == "__main__":
    # Lee 'PORT' (inyectado por Render), si no existe usa 10000 como fallback por defecto
    port = int(os.environ.get("PORT", 10000))
    
    print(f"Iniciando Gradio de forma estricta en 0.0.0.0:{port}...") 
    
    # 3. CAMBIO: Agregamos parámetros de seguridad para estabilizar Gradio en servidores en la nube
    demo.launch(
        server_name="0.0.0.0", 
        server_port=port,
        share=False,               # Desactiva túneles públicos de Gradio (chocan con Render)
        prevent_thread_lock=False  # Permite que el hilo principal retenga el servidor abierto
    )
