from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import json

app = FastAPI()

# ==========================
# CORS
# ==========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Home
# ==========================

@app.get("/")
def home():

    return {
        "message": "NeuroWatch AI Backend Running"
    }

# ==========================
# Real AI Status
# ==========================

@app.get("/status")
def get_status():

    try:

        with open(
            "ai_state.json",
            "r"
        ) as file:

            data = json.load(file)

        return data

    except Exception as e:

        return {

            "status": "ERROR",

            "threat": "LOW",

            "fps": 0,

            "detections": 0,

            "accuracy": 76.62,

            "error": str(e)
        }