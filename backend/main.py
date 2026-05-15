from fastapi import FastAPI

app = FastAPI(title="NeuroWatch AI")

@app.get("/")
def home():
    return {"message": "NeuroWatch AI Backend Running"}