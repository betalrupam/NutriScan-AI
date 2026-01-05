import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # Added for serving HTML

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Literal

load_dotenv()

# --- SCHEMAS ---
class ChemicalSchema(BaseModel):
    compoundName: Annotated[str, Field(description="Write the name of the compound")]
    Describe: Annotated[str, Field(description="Give a simple brief explanation under 200 words")]
    EffectOnBody: Annotated[str, Field(description="Effects on body in simple words, bullet points, under 150 words")]
    MostlyTargetPeople: Annotated[str, Field(description="Mention people who may be affected.")]
    rating: Annotated[float, Field(description="Rating based on effect", gt=0.5, lt=9.9)]

class Deteils(BaseModel):
    chemicals: List[ChemicalSchema]

class BasicInfo(BaseModel):
    totalCal: Annotated[Optional[str], Field(description="Total energy present in the product")]
    overView: Annotated[str, Field(description='Basic information under 200 words')]
    feedback: Literal[
        "You should eat this product regualarly",
        "You should eat this in controlled manner regularly",
        "You should eat this Occasanally",
        "You should avoid this",
        "Stay away from this!!"
    ]
    specialNote: Annotated[str, Field(description='Specific people affected')]
    rating: Annotated[float, Field(gt=0.5, lt=10.0)]

class Chemical(BaseModel):
    name: Annotated[str, Field(description='extract all the compound and chemicals')]
    conc: Optional[str]

class AllChemicals(BaseModel):
    chemicals: List[Chemical]

# --- LLM SETUP ---
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Updated to a valid model name
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)

# --- APP SETUP ---
app = FastAPI()

# Updated CORS for Deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows your hosted frontend to connect
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode()

# --- ROUTES ---

# 1. FIXED: Added a Root Route to serve your HTML or a status message
@app.get("/")
async def read_root():
    # If you have index.html in the same folder, use this:
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return {"message": "NutriScan API is Live. Please use the frontend to upload images."}

@app.post("/basicdata")
async def basicdata(file: UploadFile = File(...)):
    img_bytes = await file.read()
    base64_image = image_to_base64(img_bytes)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze the nutrition label. Return JSON only."},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )
    parser = llm2.with_structured_output(BasicInfo)
    return parser.invoke([msg])

@app.post("/specialdata")
async def specialdata(file: UploadFile = File(...)):
    img_bytes = await file.read()
    base64_image = image_to_base64(img_bytes)
    msg = HumanMessage(
        content=[
            {"type": "text", "text": "List all food chemicals + quantity"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
        ]
    )
    extract = llm2.with_structured_output(AllChemicals)
    result = extract.invoke([msg])
    chemicals_text = "\n".join([f"- {c.name} ({c.conc or 'n/a'})" for c in result.chemicals])

    prompt = f"Return JSON for these chemicals: {chemicals_text}"
    structured = llm.with_structured_output(Deteils)
    return structured.invoke(prompt)
