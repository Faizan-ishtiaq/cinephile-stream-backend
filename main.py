from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
import requests
import random
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")

class ChatRequest(BaseModel):
    message: str

MOVIE_GENRE_MAP = {
    "comedy": 35,
    "funny": 35,
    "action": 28,
    "horror": 27,
    "romance": 10749,
    "drama": 18,
    "thriller": 53,
}

TV_GENRE_MAP = {
    "comedy": 35,
    "funny": 35,
    "action": 10759,
    "horror": 9648,  # mystery, closest available
    "romance": 10749,
    "drama": 18,
    "thriller": 9648,
}

def get_genre_id(message, media_type):
    message = message.lower()
    genre_map = TV_GENRE_MAP if media_type == "tv" else MOVIE_GENRE_MAP
    for keyword, genre_id in genre_map.items():
        if keyword in message:
            return genre_id
    return 35

def detect_media_type(message):
    message = message.lower()
    series_keywords = ["series", "show", "tv show", "season"]
    if any(keyword in message for keyword in series_keywords):
        return "tv"
    return "movie"

@app.post("/recommend")
def recommend(request: ChatRequest):
    media_type = detect_media_type(request.message)
    genre_id = get_genre_id(request.message, media_type)
    random_page = random.randint(1, 5)
    tmdb_url = f"https://api.themoviedb.org/3/discover/{media_type}?api_key={TMDB_API_KEY}&with_genres={genre_id}&sort_by=vote_average.desc&vote_count.gte=200&page={random_page}"
    tmdb_response = requests.get(tmdb_url).json()
    results = tmdb_response.get("results", [])[:10]

    titles = [item.get("title") or item.get("name") for item in results]

    system_prompt = f"""You are a movie and TV series recommendation expert. 
    Here are real, currently available {media_type}s: {titles}.
    Based on the user's request, pick and recommend from THIS list only. 
    Keep it concise and friendly."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.message}
        ]
    )

    reply = response.choices[0].message.content
    return {"reply": reply}