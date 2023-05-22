from fastapi import FastAPI
app =FastAPI(title="MLOPs Basics App")

@app.get("/")
async def home():
    return "<h2>This is a sample NLP Project</h2>"