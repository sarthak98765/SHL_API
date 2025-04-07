
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from bs4 import BeautifulSoup

app = FastAPI()

# Load model and data
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
df = pd.read_csv('shl_assessments_data.csv')
corpus_embeddings = np.load('corpus_embeddings.npy')

# Pydantic model
class QueryRequest(BaseModel):
    text: str

# Main recommendation function
def recommend_assessments(user_input, top_n=10):
    query_embedding = model.encode([user_input])
    similarity_scores = cosine_similarity(query_embedding, corpus_embeddings)[0]
    top_indices = similarity_scores.argsort()[::-1][:top_n]

    results = df.loc[top_indices, [
        'Assessment Name', 'URL', 'Remote Testing', 'Adaptive/IRT', 'Duration', 'Test Type'
    ]].copy()

    results['Assessment Name'] = results.apply(
        lambda row: f"[{row['Assessment Name']}]({row['URL']})", axis=1
    )

    return results.drop(columns='URL').reset_index(drop=True)

# Optional: Extract job description text from a URL
def extract_text_from_url(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator=' ', strip=True)[:1500]
    except Exception as e:
        return f"Error while fetching URL: {str(e)}"

# POST endpoint
@app.post("/recommend/")
async def recommend_assessments_api(request: QueryRequest):
    try:
        results = recommend_assessments(request.text)

        # Replace problematic float values
        results.replace([np.inf, -np.inf], np.nan, inplace=True)
        results = results.where(pd.notnull(results), None)

        response_data = results.to_dict(orient='records')
        return JSONResponse(content={"recommendations": response_data})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=10000)
