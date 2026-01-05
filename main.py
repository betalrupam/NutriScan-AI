import os
import base64
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Annotated, List, Optional, Literal

load_dotenv()

class ChemicalSchema(BaseModel):
    compoundName: Annotated[
        str, Field(description="Write the name of the compound")
    ]

    Describe: Annotated[
        str, Field(description="Give a simple brief explanation under 200 words")
    ]

    EffectOnBody: Annotated[
        str, Field(
            description="Effects on body in simple words, bullet points, under 150 words"
        )
    ]

    MostlyTargetPeople: Annotated[
        str,
        Field(
            description=(
                "Mention people who may be affected.\n"
                "Format:\n1.Diabetic\n2.Sensitive Skin\n"
                "If none: No specific group"
            )
        )
    ]

    rating: Annotated[
        float,
        Field(description="Rating based on effect", gt=0.5, lt=9.9)
    ]


class Deteils(BaseModel):
    chemicals: List[ChemicalSchema]


class BasicInfo(BaseModel):
    totalCal: Annotated[
        Optional[str],
        Field(
            description=(
                "Total energy present in the product with quantity. "
                "If not clearly visible, estimate an approximate value."
            ),
            examples=["400 Kcal per 50 gram"]
        )
    ]

    overView: Annotated[
        str,Field(
            description='Give a basic information about the food in a simple manner under 200 words'
        )
    ]
    feedback: Literal[
        "You should eat this product regualarly",
        "You should eat this in controlled manner regularly",
        "You should eat this Occasanally",
        "You should avoid this",
        "Stay away from this!!"
    ]
    specialNote:Annotated[str,
                          Field(
                              description='Tell if some spefific kind of people can effect',examples=['1.Overweight people 2.Diabatic people']
                          )]
    rating: Annotated[float, Field(gt=0.5, lt=10.0)]
    


class Chemical(BaseModel):
    name:Annotated[
        str,Field(
            description='extract all the compound and chemicals that present in the given data....remember that energy is not a compound'
        )
    ]
    conc: Optional[str]


class AllChemicals(BaseModel):
    chemicals: List[Chemical]

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0.2
)

llm2 = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def image_to_base64(file_bytes):
    return base64.b64encode(file_bytes).decode()


@app.post("/basicdata")
async def basicdata(file: UploadFile = File(...)):

    img_bytes = await file.read()
    base64_image = image_to_base64(img_bytes)

    msg = HumanMessage(
        content=[
            {"type": "text", "text": "Analyze the nutrition label. Return JSON only."},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            },
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
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
            },
        ]
    )

    extract = llm2.with_structured_output(AllChemicals)
    result = extract.invoke([msg])

    chemicals_text = "\n".join(
        f"- {c.name} ({c.conc or 'quantity not specified'})"
        for c in result.chemicals
    )

    prompt = f"""
You are a nutrition expert.

Return VALID JSON strictly matching this schema:

{{
  "chemicals": [
    {{
      "compoundName": string,
      "Describe": string (≤200 words),
      "EffectOnBody": string (bullet points, ≤150 words),
      "MostlyTargetPeople": string (numbered list OR "No specific group"),
      "rating": float (0.5–9.9)
    }}
  ]
}}

Rules:
- JSON ONLY
- No extra text
- Simple language (10-year-old)
- Effects MUST be bullet points

CHEMICAL LIST:
{chemicals_text}
"""


    structured = llm.with_structured_output(Deteils)
    return structured.invoke(prompt)
