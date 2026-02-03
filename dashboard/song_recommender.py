"""
Song Recommendation System
Uses GenAI (Gemini) to recommend songs based on mood and weather
Falls back to curated playlists with daily rotation when API unavailable
"""

import os
import json
import requests
import random
from datetime import datetime


def get_gemini_api_key():
    """Get Gemini API key from environment"""
    # Try different possible locations
    api_key = os.environ.get('GEMINI_API_KEY')

    if not api_key:
        # Try loading from airflow-playground .env
        env_path = '/Users/tolgasabanoglu/Desktop/github/airflow-playground/.env'
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    if line.startswith('GEMINI_API_KEY='):
                        api_key = line.split('=', 1)[1].strip().strip('"\'')
                        break

    return api_key


def get_fallback_recommendations(mood_profile, weather_temp):
    """Curated playlists for when API is unavailable - with daily rotation for variety"""
    playlists = {
        "cozy_indoor": [
            {"title": "Holocene", "artist": "Bon Iver", "reason": "Warm, introspective melodies perfect for cold, reflective moments indoors"},
            {"title": "To Build a Home", "artist": "The Cinematic Orchestra", "reason": "Emotional depth that creates a cozy, contemplative atmosphere"},
            {"title": "Skinny Love", "artist": "Bon Iver", "reason": "Intimate acoustic sound ideal for quiet, comfortable spaces"},
            {"title": "Flume", "artist": "Bon Iver", "reason": "Delicate textures create a peaceful indoor sanctuary"},
            {"title": "The Night We Met", "artist": "Lord Huron", "reason": "Nostalgic warmth perfect for cozy introspection"},
            {"title": "Kathleen", "artist": "Josh Ritter", "reason": "Gentle storytelling for comfortable, reflective moments"},
            {"title": "Bloodbank", "artist": "Bon Iver", "reason": "Hushed intimacy ideal for quiet indoor comfort"},
            {"title": "For Emma", "artist": "Bon Iver", "reason": "Raw emotion wrapped in cozy acoustic warmth"},
            {"title": "Such Great Heights", "artist": "Iron & Wine", "reason": "Soft, tender cover perfect for peaceful moments"}
        ],
        "green_nature": [
            {"title": "Banana Pancakes", "artist": "Jack Johnson", "reason": "Laid-back, nature-inspired vibes perfect for relaxed outdoor moments"},
            {"title": "Here Comes the Sun", "artist": "The Beatles", "reason": "Uplifting and natural, captures the essence of green spaces"},
            {"title": "Big Sur", "artist": "The Thrills", "reason": "Breezy, carefree energy that matches peaceful natural settings"},
            {"title": "Island in the Sun", "artist": "Weezer", "reason": "Sunny, outdoor vibes for green space relaxation"},
            {"title": "The Wolf", "artist": "Mumford & Sons", "reason": "Folk energy that connects with natural surroundings"},
            {"title": "Ends of the Earth", "artist": "Lord Huron", "reason": "Wandering melodies perfect for outdoor exploration"},
            {"title": "Rivers and Roads", "artist": "The Head and the Heart", "reason": "Natural imagery and peaceful acoustic warmth"},
            {"title": "Harvest Moon", "artist": "Neil Young", "reason": "Timeless folk perfect for green, open spaces"},
            {"title": "Garden Song", "artist": "Phoebe Bridgers", "reason": "Nature-inspired introspection with gentle acoustics"}
        ],
        "buzz_urban": [
            {"title": "Electric Feel", "artist": "MGMT", "reason": "High-energy synths perfect for vibrant city nightlife"},
            {"title": "Digital Love", "artist": "Daft Punk", "reason": "Pulsing beats that match urban energy and social scenes"},
            {"title": "Midnight City", "artist": "M83", "reason": "Epic, driving rhythm perfect for bustling metropolitan vibes"},
            {"title": "Pumped Up Kicks", "artist": "Foster the People", "reason": "Catchy beats for energetic urban movement"},
            {"title": "Get Lucky", "artist": "Daft Punk", "reason": "Groovy, social energy for vibrant nightlife"},
            {"title": "Feel It Still", "artist": "Portugal. The Man", "reason": "Modern funk perfect for urban social scenes"},
            {"title": "Tongue Tied", "artist": "Grouplove", "reason": "High-energy anthems for bustling city vibes"},
            {"title": "Walking on a Dream", "artist": "Empire of the Sun", "reason": "Euphoric synths matching metropolitan energy"},
            {"title": "Safe and Sound", "artist": "Capital Cities", "reason": "Upbeat urban pop for social, energetic moments"}
        ],
        "rainy_retreat": [
            {"title": "Let It Be", "artist": "The Beatles", "reason": "Calming and reassuring, perfect for weathering a storm"},
            {"title": "The Night We Met", "artist": "Lord Huron", "reason": "Nostalgic and cozy, ideal for rainy day introspection"},
            {"title": "Champagne Supernova", "artist": "Oasis", "reason": "Dreamy and atmospheric, matches the mood of rain outside"},
            {"title": "Fake Plastic Trees", "artist": "Radiohead", "reason": "Melancholic beauty for contemplative rainy moments"},
            {"title": "Pursued by a Bear", "artist": "The Tallest Man on Earth", "reason": "Intimate folk for sheltered reflection during storms"},
            {"title": "Falling Slowly", "artist": "Glen Hansard", "reason": "Gentle piano matching the rhythm of rain"},
            {"title": "Mad World", "artist": "Gary Jules", "reason": "Contemplative melancholy perfect for rainy introspection"},
            {"title": "The Blower's Daughter", "artist": "Damien Rice", "reason": "Emotional depth for sheltered, rainy moments"},
            {"title": "To Build a Home", "artist": "The Cinematic Orchestra", "reason": "Cinematic warmth for cozy rain retreats"}
        ],
        "cozy_recharge": [
            {"title": "Breathe Me", "artist": "Sia", "reason": "Gentle and restorative, perfect for energy recovery"},
            {"title": "Samson", "artist": "Regina Spektor", "reason": "Soft piano melodies that support quiet recharging"},
            {"title": "Mad World", "artist": "Gary Jules", "reason": "Contemplative and calm, ideal for rest and reflection"},
            {"title": "Cosmic Love", "artist": "Florence + The Machine", "reason": "Ethereal beauty for peaceful energy restoration"},
            {"title": "Skinny Love", "artist": "Birdy", "reason": "Delicate piano cover perfect for gentle recovery"},
            {"title": "Turning Page", "artist": "Sleeping At Last", "reason": "Soothing melodies that support recharging"},
            {"title": "Holocene", "artist": "Bon Iver", "reason": "Peaceful atmospheres for restful recovery"},
            {"title": "The Call", "artist": "Regina Spektor", "reason": "Soft, restorative tones for energy renewal"},
            {"title": "Eyes on Fire", "artist": "Blue Foundation", "reason": "Ambient calm perfect for quiet recharge"}
        ],
        "balanced": [
            {"title": "Ho Hey", "artist": "The Lumineers", "reason": "Upbeat yet mellow, perfect for a balanced, flexible mood"},
            {"title": "Home", "artist": "Edward Sharpe & The Magnetic Zeros", "reason": "Feel-good vibes that work for various activities"},
            {"title": "Budapest", "artist": "George Ezra", "reason": "Cheerful and moderate energy, suits balanced states"},
            {"title": "Riptide", "artist": "Vance Joy", "reason": "Light and catchy, perfect for balanced mood"},
            {"title": "Little Talks", "artist": "Of Monsters and Men", "reason": "Uplifting folk-pop for flexible activities"},
            {"title": "Some Nights", "artist": "fun.", "reason": "Dynamic energy that adapts to various moods"},
            {"title": "Dog Days Are Over", "artist": "Florence + The Machine", "reason": "Joyful energy for balanced, positive moments"},
            {"title": "Mr. Brightside", "artist": "The Killers", "reason": "Anthemic energy perfect for any activity"},
            {"title": "I Will Wait", "artist": "Mumford & Sons", "reason": "Building energy that suits balanced states"}
        ]
    }

    # Get songs for mood profile
    available_songs = playlists.get(mood_profile, playlists["balanced"])

    # Use daily seed for consistent rotation throughout the day
    date_seed = int(datetime.now().date().toordinal())
    random.seed(date_seed)

    # Randomly select 3 songs from the available pool
    selected_songs = random.sample(available_songs, min(3, len(available_songs)))

    return selected_songs


def get_song_recommendations(mood_profile, stress, sleep_hours, weather_temp, weather_precip):
    """
    Get song recommendations based on mood and weather using Gemini API

    Args:
        mood_profile: Mood type (cozy_indoor, green_nature, buzz_urban, etc.)
        stress: Average stress level (0-100)
        sleep_hours: Average sleep hours
        weather_temp: Current temperature (°C)
        weather_precip: Current precipitation (mm)

    Returns:
        list: List of song recommendation dicts
    """
    api_key = get_gemini_api_key()

    # Check if API key is valid (not placeholder)
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE" or len(api_key) < 20:
        # Use fallback curated playlist
        return get_fallback_recommendations(mood_profile, weather_temp)

    # Map mood to descriptive text
    mood_descriptions = {
        "cozy_indoor": "cozy and warm, seeking comfort and relaxation indoors",
        "green_nature": "seeking natural, peaceful, and restorative vibes",
        "buzz_urban": "energetic, social, and ready for vibrant activity",
        "rainy_retreat": "sheltered and calm, embracing cozy ambiance during bad weather",
        "cozy_recharge": "in recovery mode, needing quiet and restorative energy",
        "balanced": "in a balanced state, open to moderate activity"
    }

    mood_desc = mood_descriptions.get(mood_profile, "in a neutral mood")

    # Build weather description
    if weather_temp < 5:
        weather_desc = f"cold weather ({weather_temp:.1f}°C)"
    elif weather_temp > 20:
        weather_desc = f"warm weather ({weather_temp:.1f}°C)"
    else:
        weather_desc = f"mild weather ({weather_temp:.1f}°C)"

    if weather_precip > 5:
        weather_desc += f" with rain ({weather_precip:.1f}mm)"

    # Create prompt for Gemini
    prompt = f"""You are a music recommendation expert. Based on the following context, recommend exactly 3 songs that would perfectly match this moment.

Context:
- Current mood: {mood_desc}
- Stress level: {stress:.1f}/100
- Sleep quality: {sleep_hours:.1f} hours (average)
- Weather: {weather_desc}

Please recommend 3 songs that would be perfect for this situation. For each song, provide:
1. Song title
2. Artist name
3. A brief reason (1 sentence) why this song fits the mood and weather

Format your response as a JSON array like this:
[
  {{"title": "Song Name", "artist": "Artist Name", "reason": "Brief reason why it fits"}},
  {{"title": "Song Name 2", "artist": "Artist Name 2", "reason": "Brief reason why it fits"}},
  {{"title": "Song Name 3", "artist": "Artist Name 3", "reason": "Brief reason why it fits"}}
]

Only return the JSON array, no other text."""

    try:
        # Call Gemini API
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        headers = {
            "Content-Type": "application/json"
        }

        data = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.9,
                "maxOutputTokens": 1000
            }
        }

        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()

        result = response.json()

        # Extract text from response
        if 'candidates' in result and len(result['candidates']) > 0:
            text = result['candidates'][0]['content']['parts'][0]['text']

            # Try to parse JSON from the response
            # Sometimes Gemini wraps JSON in markdown code blocks
            if '```json' in text:
                text = text.split('```json')[1].split('```')[0].strip()
            elif '```' in text:
                text = text.split('```')[1].split('```')[0].strip()

            songs = json.loads(text)
            return songs
        else:
            return [{
                "title": "Error",
                "artist": "N/A",
                "reason": "No response from Gemini API"
            }]

    except json.JSONDecodeError as e:
        return [{
            "title": "Parsing Error",
            "artist": "N/A",
            "reason": f"Could not parse Gemini response: {str(e)}"
        }]
    except requests.exceptions.RequestException as e:
        return [{
            "title": "API Error",
            "artist": "N/A",
            "reason": f"Could not connect to Gemini API: {str(e)}"
        }]
    except Exception as e:
        return [{
            "title": "Error",
            "artist": "N/A",
            "reason": f"Unexpected error: {str(e)}"
        }]
