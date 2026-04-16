#!/usr/bin/env python3
"""
Fetch major world cities with country, region, and metadata.

Data sources:
1. REST Countries API (https://restcountries.com/)
   — countries, capitals, regions, capital coordinates
2. Wikidata API (https://www.wikidata.org/)
   — city population, founding year, coordinates
3. GaWC 2020 classification (hardcoded)
   — business activity levels for major world cities

Output: data/cities.csv with columns:
  city, country, region, lat, lon, population, founded, business_activity

Usage:
    python scripts/fetch_cities.py
"""

import csv
import time
from pathlib import Path

import requests

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "cities.csv"

COUNTRIES_API_URL = "https://restcountries.com/v3.1/all"
WIKIDATA_SEARCH_URL = "https://www.wikidata.org/w/api.php"
WIKIDATA_ENTITY_URL = "https://www.wikidata.org/wiki/Special:EntityData/{}.json"

REQUEST_HEADERS = {"User-Agent": "CityFetcher/1.0 (educational project)"}

# Rate limit: seconds between Wikidata requests
WIKIDATA_DELAY = 0.5


# ---------------------------------------------------------------------------
# GaWC 2020 business-activity classification
# https://en.wikipedia.org/wiki/Globalization_and_World_Cities_Research_Network
#
# Categories (high → low):
#   Alpha++  Alpha+  Alpha  Alpha-
#   Beta+    Beta    Beta-
#   Gamma+   Gamma   Gamma-
#   Sufficiency  High Sufficiency
# ---------------------------------------------------------------------------

GAWC_2020: dict[str, str] = {
    # Alpha++
    "London": "Alpha++",
    "New York City": "Alpha++",
    # Alpha+
    "Beijing": "Alpha+",
    "Dubai": "Alpha+",
    "Hong Kong": "Alpha+",
    "Paris": "Alpha+",
    "Shanghai": "Alpha+",
    "Singapore": "Alpha+",
    "Tokyo": "Alpha+",
    # Alpha
    "Chicago": "Alpha",
    "Guangzhou": "Alpha",
    "Istanbul": "Alpha",
    "Johannesburg": "Alpha",
    "Kuala Lumpur": "Alpha",
    "Los Angeles": "Alpha",
    "Madrid": "Alpha",
    "Moscow": "Alpha",
    "Mumbai": "Alpha",
    "São Paulo": "Alpha",
    "Shenzhen": "Alpha",
    "Toronto": "Alpha",
    # Alpha-
    "Amsterdam": "Alpha-",
    "Bangkok": "Alpha-",
    "Barcelona": "Alpha-",
    "Beijing": "Alpha-",
    "Berlin": "Alpha-",
    "Bogotá": "Alpha-",
    "Brussels": "Alpha-",
    "Buenos Aires": "Alpha-",
    "Cairo": "Alpha-",
    "Casablanca": "Alpha-",  
    "Dallas": "Alpha-",
    "Delhi": "Alpha-",
    "Doha": "Alpha-",
    "Dublin": "Alpha-",
    "Frankfurt": "Alpha-",
    "Houston": "Alpha-",
    "Jakarta": "Alpha-",
    "Kolkata": "Alpha-",
    "Lima": "Alpha-",
    "Lisbon": "Alpha-",
    "Manila": "Alpha-",
    "Mexico City": "Alpha-",
    "Milan": "Alpha-",
    "Monterrey": "Alpha-",
    "Munich": "Alpha-",
    "Nairobi": "Alpha-",
    "New Delhi": "Alpha-",
    "Osaka": "Alpha-",
    "Panama City": "Alpha-",
    "Riyadh": "Alpha-",
    "Rome": "Alpha-",
    "San Francisco": "Alpha-",
    "Santiago": "Alpha-",
    "São Paulo": "Alpha-",
    "Seoul": "Alpha-",
    "Stockholm": "Alpha-",
    "Taipei": "Alpha-",
    "Tel Aviv": "Alpha-",
    "Vienna": "Alpha-",
    "Warsaw": "Alpha-",
    "Washington, D.C.": "Alpha-",
    "Zurich": "Alpha-",
    # Beta+
    "Atlanta": "Beta+",
    "Austin": "Beta+",
    "Bengaluru": "Beta+",
    "Brisbane": "Beta+",
    "Budapest": "Beta+",
    "Cape Town": "Beta+",
    "Caracas": "Beta+",
    "Chennai": "Beta+",
    "Copenhagen": "Beta+",
    "Denver": "Beta+",
    "Guatemala City": "Beta+",
    "Hamburg": "Beta+",
    "Helsinki": "Beta+",
    "Ho Chi Minh City": "Beta+",
    "Kuwait City": "Beta+",
    "Miami": "Beta+",
    "Minneapolis": "Beta+",
    "Montevideo": "Beta+",
    "Nanjing": "Beta+",
    "Paris": "Beta+",
    "Perth": "Beta+",
    "Philadelphia": "Beta+",
    "Phoenix": "Beta+",
    "Pretoria": "Beta+",
    "Qatar (Doha)": "Beta+",
    "San Diego": "Beta+",
    "Santiago": "Beta+",
    "Seattle": "Beta+",
    "Stuttgart": "Beta+",
    "Tegucigalpa": "Beta+",
    "The Hague": "Beta+",
    "Tianjin": "Beta+",
    "Vancouver": "Beta+",
    "Washington, D.C.": "Beta+",
    "Wellington": "Beta+",
    # Beta
    "Abu Dhabi": "Beta",
    "Accra": "Beta",
    "Baku": "Beta",
    "Bucharest": "Beta",
    "Chengdu": "Beta",
    "Colombo": "Beta",
    "Dakar": "Beta",
    "Düsseldorf": "Beta",
    "Edinburgh": "Beta",
    "Guayaquil": "Beta",
    "Harare": "Beta",
    "Hangzhou": "Beta",
    "Havana": "Beta",
    "Kunming": "Beta",
    "Lausanne": "Beta",
    "Leeds": "Beta",
    "Lyon": "Beta",
    "Manama": "Beta",
    "Marseille": "Beta",
    "Melbourne": "Beta",
    "Montreal": "Beta",
    "Muscat": "Beta",
    "Nairobi": "Beta",
    "Oslo": "Beta",
    "Prague": "Beta",
    "Qingdao": "Beta",
    "Quito": "Beta",
    "Riga": "Beta",
    "Rio de Janeiro": "Beta",
    "San Jose": "Beta",
    "Shenzhen": "Beta",
    "Sofia": "Beta",
    "St. Petersburg": "Beta",
    "Tashkent": "Beta",
    "Tbilisi": "Beta",
    "Tehran": "Beta",
    "Valencia": "Beta",
    "Vilnius": "Beta",
    "Wuhan": "Beta",
    "Xi'an": "Beta",
    # Beta-
    "Algiers": "Beta-",
    "Almaty": "Beta-",
    "Antwerp": "Beta-",
    "Auckland": "Beta-",
    "Belgrade": "Beta-",
    "Birmingham": "Beta-",
    "Bogotá": "Beta-",
    "Brisbane": "Beta-",
    "Bucharest": "Beta-",
    "Cairo": "Beta-",
    "Changsha": "Beta-",
    "Chongqing": "Beta-",
    "Dalian": "Beta-",
    "Detroit": "Beta-",
    "Dhaka": "Beta-",
    "Dongguan": "Beta-",
    "Fukuoka": "Beta-",
    "Genoa": "Beta-",
    "Glasgow": "Beta-",
    "Gothenburg": "Beta-",
    "Hanoi": "Beta-",
    "Kaohsiung": "Beta-",
    "Karachi": "Beta-",
    "Kiev": "Beta-",
    "Kinshasa": "Beta-",
    "Lagos": "Beta-",
    "Lahore": "Beta-",
    "Lille": "Beta-",
    "Lusaka": "Beta-",
    "Medellín": "Beta-",
    "Montpellier": "Beta-",
    "Nagoya": "Beta-",
    "Naples": "Beta-",
    "Ningbo": "Beta-",
    "Osaka": "Beta-",
    "Portland": "Beta-",
    "Porto": "Beta-",
    "Rotterdam": "Beta-",
    "Shenyang": "Beta-",
    "St. Louis": "Beta-",
    "Surabaya": "Beta-",
    "Taiyuan": "Beta-",
    "Tampa": "Beta-",
    "Turin": "Beta-",
    "Ulsan": "Beta-",
    "Wenzhou": "Beta-",
    "Xiamen": "Beta-",
    "Yangon": "Beta-",
    "Zhengzhou": "Beta-",
    # Gamma+
    "Adelaide": "Gamma+",
    "Ankara": "Gamma+",
    "Athens": "Gamma+",
    "Baltimore": "Gamma+",
    "Bangkok": "Gamma+",
    "Belo Horizonte": "Gamma+",
    "Charlotte": "Gamma+",
    "Cologne": "Gamma+",
    "Curitiba": "Gamma+",
    "Dar es Salaam": "Gamma+",
    "Douala": "Gamma+",
    "Durban": "Gamma+",
    "Gwangju": "Gamma+",
    "Hefei": "Gamma+",
    "Indianapolis": "Gamma+",
    "Incheon": "Gamma+",
    "Izmir": "Gamma+",
    "Jeddah": "Gamma+",
    "Johor Bahru": "Gamma+",
    "Kansas City": "Gamma+",
    "Košice": "Gamma+",
    "Krakow": "Gamma+",
    "Luanda": "Gamma+",
    "Lubumbashi": "Gamma+",
    "Malmö": "Gamma+",
    "Maputo": "Gamma+",
    "Minsk": "Gamma+",
    "Nashville": "Gamma+",
    "Nicosia": "Gamma+",
    "Orlando": "Gamma+",
    "Pittsburgh": "Gamma+",
    "Portland": "Gamma+",
    "Raleigh": "Gamma+",
    "Riverside": "Gamma+",
    "Rostov-on-Don": "Gamma+",
    "Sacramento": "Gamma+",
    "San Salvador": "Gamma+",
    "Santo Domingo": "Gamma+",
    "Seville": "Gamma+",
    "Shijiazhuang": "Gamma+",
    "Sucre": "Gamma+",
    "Taichung": "Gamma+",
    "Tampa": "Gamma+",
    "Tbilisi": "Gamma+",
    "Tel Aviv": "Gamma+",
    "Thessaloniki": "Gamma+",
    "Toulouse": "Gamma+",
    "Tunis": "Gamma+",
    "Utrec": "Gamma+",
    "Wrocław": "Gamma+",
    "Yekaterinburg": "Gamma+",
    # Gamma
    "Abidjan": "Gamma",
    "Addis Ababa": "Gamma",
    "Barcelona": "Gamma",
    "Belfast": "Gamma",
    "Bogota": "Gamma",
    "Bucharest": "Gamma",
    "Cali": "Gamma",
    "Cape Town": "Gamma",
    "Changchun": "Gamma",
    "Chittagong": "Gamma",
    "Cincinnati": "Gamma",
    "Dakar": "Gamma",
    "Dammam": "Gamma",
    "Dortmund": "Gamma",
    "Dubai": "Gamma",
    "Essen": "Gamma",
    "Foshan": "Gamma",
    "Guadalajara": "Gamma",
    "Guiyang": "Gamma",
    "Haikou": "Gamma",
    "Harbin": "Gamma",
    "Harrisburg": "Gamma",
    "Hofuf": "Gamma",
    "Jacksonville": "Gamma",
    "Jinan": "Gamma",
    "Johor Bahru": "Gamma",
    "Kampala": "Gamma",
    "Kazan": "Gamma",
    "Kochi": "Gamma",
    "Kolkata": "Gamma",
    "Kuala Lumpur": "Gamma",
    "Las Vegas": "Gamma",
    "Leeds": "Gamma",
    "Leon": "Gamma",
    "Lusaka": "Gamma",
    "Maputo": "Gamma",
    "Marrakesh": "Gamma",
    "Medellín": "Gamma",
    "Milwaukee": "Gamma",
    "Monterrey": "Gamma",
    "Nantes": "Gamma",
    "New Orleans": "Gamma",
    "Oklahoma City": "Gamma",
    "Palermo": "Gamma",
    "Perugia": "Gamma",
    "Phnom Penh": "Gamma",
    "Phuket": "Gamma",
    "Port Moresby": "Gamma",
    "Porto Alegre": "Gamma",
    "Pyongyang": "Gamma",
    "Recife": "Gamma",
    "Reykjavik": "Gamma",
    "Riga": "Gamma",
    "Rosario": "Gamma",
    "Sana'a": "Gamma",
    "Saskatoon": "Gamma",
    "Sevilla": "Gamma",
    "Strasbourg": "Gamma",
    "Stuttgart": "Gamma",
    "Sucre": "Gamma",
    "Tbilisi": "Gamma",
    "Thessaloniki": "Gamma",
    "Tijuana": "Gamma",
    "Turku": "Gamma",
    "Valencia": "Gamma",
    "Vilnius": "Gamma",
    "Wuhan": "Gamma",
    "Zagreb": "Gamma",
    "Zurich": "Gamma",
    # Gamma-
    "Ahmedabad": "Gamma-",
    "Astana": "Gamma-",
    "Baku": "Gamma-",
    "Bhopal": "Gamma-",
    "Birmingham": "Gamma-",
    "Bogota": "Gamma-",
    "Bucharest": "Gamma-",
    "Busan": "Gamma-",
    "Cairo": "Gamma-",
    "Caracas": "Gamma-",
    "Changsha": "Gamma-",
    "Chongqing": "Gamma-",
    "Cleveland": "Gamma-",
    "Columbus": "Gamma-",
    "Dalian": "Gamma-",
    "Denver": "Gamma-",
    "Detroit": "Gamma-",
    "Dhaka": "Gamma-",
    "Dongguan": "Gamma-",
    "Fukuoka": "Gamma-",
    "Genoa": "Gamma-",
    "Glasgow": "Gamma-",
    "Gothenburg": "Gamma-",
    "Guayaquil": "Gamma-",
    "Hanoi": "Gamma-",
    "Harare": "Gamma-",
    "Havana": "Gamma-",
    "Ho Chi Minh City": "Gamma-",
    "Indianapolis": "Gamma-",
    "Jaipur": "Gamma-",
    "Jeddah": "Gamma-",
    "Karachi": "Gamma-",
    "Kazan": "Gamma-",
    "Kiev": "Gamma-",
    "Kochi": "Gamma-",
    "Krakow": "Gamma-",
    "Kunming": "Gamma-",
    "Lahore": "Gamma-",
    "Lille": "Gamma-",
    "Luanda": "Gamma-",
    "Lusaka": "Gamma-",
    "Medellín": "Gamma-",
    "Melbourne": "Gamma-",
    "Montevideo": "Gamma-",
    "Montpellier": "Gamma-",
    "Nagoya": "Gamma-",
    "Naples": "Gamma-",
    "Ningbo": "Gamma-",
    "Oslo": "Gamma-",
    "Porto": "Gamma-",
    "Portland": "Gamma-",
    "Prague": "Gamma-",
    "Quito": "Gamma-",
    "Riga": "Gamma-",
    "Rio de Janeiro": "Gamma-",
    "Rotterdam": "Gamma-",
    "San Jose": "Gamma-",
    "Shenyang": "Gamma-",
    "Sofia": "Gamma-",
    "St. Petersburg": "Gamma-",
    "St. Louis": "Gamma-",
    "Surabaya": "Gamma-",
    "Taichung": "Gamma-",
    "Taiyuan": "Gamma-",
    "Tampa": "Gamma-",
    "Tashkent": "Gamma-",
    "Tbilisi": "Gamma-",
    "Tehran": "Gamma-",
    "Turin": "Gamma-",
    "Ulsan": "Gamma-",
    "Valencia": "Gamma-",
    "Vilnius": "Gamma-",
    "Wenzhou": "Gamma-",
    "Xiamen": "Gamma-",
    "Yangon": "Gamma-",
    "Zhengzhou": "Gamma-",
    # High Sufficiency
    "Cordoba": "High Sufficiency",
    "Dakar": "High Sufficiency",
    "Durban": "High Sufficiency",
    "Edinburgh": "High Sufficiency",
    "Gaborone": "High Sufficiency",
    "Glasgow": "High Sufficiency",
    "Gothenburg": "High Sufficiency",
    "Guangzhou": "High Sufficiency",
    "Halifax": "High Sufficiency",
    "Harrisburg": "High Sufficiency",
    "Indianapolis": "High Sufficiency",
    "Kazan": "High Sufficiency",
    "Lausanne": "High Sufficiency",
    "Leeds": "High Sufficiency",
    "Leipzig": "High Sufficiency",
    "Lille": "High Sufficiency",
    "Linz": "High Sufficiency",
    "Ljubljana": "High Sufficiency",
    "Luxembourg": "High Sufficiency",
    "Lyon": "High Sufficiency",
    "Malmo": "High Sufficiency",
    "Mannheim": "High Sufficiency",
    "Marseille": "High Sufficiency",
    "Nantes": "High Sufficiency",
    "Newcastle upon Tyne": "High Sufficiency",
    "Nice": "High Sufficiency",
    "Nicosia": "High Sufficiency",
    "Oporto": "High Sufficiency",
    "Oslo": "High Sufficiency",
    "Ostrava": "High Sufficiency",
    "Parma": "High Sufficiency",
    "Poznan": "High Sufficiency",
    "Prague": "High Sufficiency",
    "Quebec": "High Sufficiency",
    "Reykjavik": "High Sufficiency",
    "Riga": "High Sufficiency",
    "Rotterdam": "High Sufficiency",
    "Saarbrucken": "High Sufficiency",
    "San Jose": "High Sufficiency",
    "Sevilla": "High Sufficiency",
    "Sheffield": "High Sufficiency",
    "Skopje": "High Sufficiency",
    "Sofia": "High Sufficiency",
    "Stavanger": "High Sufficiency",
    "Strasbourg": "High Sufficiency",
    "Stuttgart": "High Sufficiency",
    "Taipei": "High Sufficiency",
    "Tallinn": "High Sufficiency",
    "Tartu": "High Sufficiency",
    "Toulouse": "High Sufficiency",
    "Turku": "High Sufficiency",
    "Utrecht": "High Sufficiency",
    "Valencia": "High Sufficiency",
    "Vienna": "High Sufficiency",
    "Vilnius": "High Sufficiency",
    "Wrocław": "High Sufficiency",
    "Zagreb": "High Sufficiency",
    "Zurich": "High Sufficiency",
}

# For cities not in GaWC, derive business_activity from population tiers
POPULATION_TIERS = [
    (10_000_000, "Alpha"),
    (5_000_000, "Alpha-"),
    (2_000_000, "Beta"),
    (1_000_000, "Beta-"),
    (500_000, "Gamma"),
    (100_000, "Gamma-"),
    (50_000, "Sufficiency"),
    (0, "High Sufficiency"),
]


def classify_business_activity(city: str, population: int | None) -> str:
    """Return GaWC business activity classification for a city.
    Falls back to population-based heuristic for cities not in GaWC."""
    # Try exact match first
    if city in GAWC_2020:
        return GAWC_2020[city]

    # Try matching without special chars / diacritics (simplified)
    city_lower = city.lower()
    for gawc_city, classification in GAWC_2020.items():
        if city_lower == gawc_city.lower():
            return classification

    # Fallback: population-based heuristic
    if population is not None and population > 0:
        for threshold, label in POPULATION_TIERS:
            if population >= threshold:
                return label

    return "Sufficiency"


# ---------------------------------------------------------------------------
# Country name normalization
# ---------------------------------------------------------------------------

COUNTRY_ALIASES = {
    "drc": "DR Congo",
    "united states": "United States",
    "united states of america": "United States",
    "usa": "United States",
    "uk": "United Kingdom",
    "united kingdom": "United Kingdom",
    "south korea": "South Korea",
    "republic of korea": "South Korea",
    "north korea": "North Korea",
    "democratic people's republic of korea": "North Korea",
    "russia": "Russia",
    "russian federation": "Russia",
    "china": "China",
    "people's republic of china": "China",
    "vietnam": "Vietnam",
    "socialist republic of vietnam": "Vietnam",
    "iran": "Iran",
    "islamic republic of iran": "Iran",
    "congo": "Republic of the Congo",
    "republic of the congo": "Republic of the Congo",
}

SKIP_TERRITORIES = {
    "united states minor outlying islands",
    "falkland islands (malvinas)",
    "french southern territories",
}


def normalize_country(country: str) -> str:
    return COUNTRY_ALIASES.get(country.lower().strip(), country)


# ---------------------------------------------------------------------------
# Data source 1: REST Countries API
# ---------------------------------------------------------------------------

def fetch_countries() -> list[dict]:
    """Fetch countries with capitals, regions, and capital coordinates."""
    print("Fetching countries from REST Countries API...")

    # Request capitalInfo for capital coordinates
    fields = "name,region,capital,capitalInfo"
    resp = requests.get(
        COUNTRIES_API_URL, params={"fields": fields}, timeout=30,
        headers=REQUEST_HEADERS,
    )
    resp.raise_for_status()

    countries = resp.json()
    print(f"  Retrieved {len(countries)} countries")

    records = []
    for country in countries:
        name = country.get("name", {}).get("common", "")
        region = country.get("region", "Unknown")
        capital_list = country.get("capital", [])
        capital = capital_list[0] if capital_list else ""

        # Capital coordinates from REST Countries
        capital_latlng = country.get("capitalInfo", {}).get("latlng", [])
        lat = round(capital_latlng[0], 4) if len(capital_latlng) >= 1 else None
        lon = round(capital_latlng[1], 4) if len(capital_latlng) >= 2 else None

        if name and capital:
            records.append({
                "city": capital,
                "country": name,
                "region": region,
                "lat": lat,
                "lon": lon,
            })

    print(f"  {len(records)} countries have capital cities")
    return records


# ---------------------------------------------------------------------------
# Data source 2: Major non-capital cities
# ---------------------------------------------------------------------------

MAJOR_NON_CAPITAL_CITIES = [
    ("New York City", "United States", "Americas", 40.7128, -74.0060),
    ("Los Angeles", "United States", "Americas", 34.0522, -118.2437),
    ("Chicago", "United States", "Americas", 41.8781, -87.6298),
    ("Houston", "United States", "Americas", 29.7604, -95.3698),
    ("Toronto", "Canada", "Americas", 43.6532, -79.3832),
    ("Mumbai", "India", "Asia", 19.0760, 72.8777),
    ("Kolkata", "India", "Asia", 22.5726, 88.3639),
    ("Bengaluru", "India", "Asia", 12.9716, 77.5946),
    ("Chennai", "India", "Asia", 13.0827, 80.2707),
    ("Delhi", "India", "Asia", 28.7041, 77.1025),
    ("Shanghai", "China", "Asia", 31.2304, 121.4737),
    ("Guangzhou", "China", "Asia", 23.1291, 113.2644),
    ("Shenzhen", "China", "Asia", 22.5431, 114.0579),
    ("Osaka", "Japan", "Asia", 34.6937, 135.5023),
    ("Yokohama", "Japan", "Asia", 35.4437, 139.6385),
    ("Istanbul", "Turkey", "Asia", 41.0082, 28.9784),
    ("São Paulo", "Brazil", "Americas", -23.5505, -46.6333),
    ("Rio de Janeiro", "Brazil", "Americas", -22.9068, -43.1729),
    ("Sydney", "Australia", "Oceania", -33.8688, 151.2093),
    ("Melbourne", "Australia", "Oceania", -37.8136, 144.9631),
    ("Auckland", "New Zealand", "Oceania", -36.8485, 174.7633),
    ("Johannesburg", "South Africa", "Africa", -26.2041, 28.0473),
    ("Casablanca", "Morocco", "Africa", 33.5731, -7.5898),
]


def fetch_major_non_capital_cities() -> list[dict]:
    """Return curated list of major non-capital cities with known coordinates."""
    print(f"Adding {len(MAJOR_NON_CAPITAL_CITIES)} well-known major cities")
    return [
        {
            "city": city,
            "country": country,
            "region": region,
            "lat": lat,
            "lon": lon,
        }
        for city, country, region, lat, lon in MAJOR_NON_CAPITAL_CITIES
    ]


# ---------------------------------------------------------------------------
# Merge & deduplicate
# ---------------------------------------------------------------------------

def merge_city_data(capitals: list[dict], major_cities: list[dict]) -> list[dict]:
    """Merge capitals and major cities, avoiding duplicates."""
    seen = set()
    records = []

    for record in capitals:
        if record["region"] == "Antarctic":
            continue
        if record["country"].lower() in SKIP_TERRITORIES:
            continue
        city = record["city"]
        country = normalize_country(record["country"])
        key = (city.lower(), country.lower())
        if key not in seen:
            seen.add(key)
            records.append({
                "city": city,
                "country": country,
                "region": record["region"],
                "lat": record.get("lat"),
                "lon": record.get("lon"),
            })

    for city_data in major_cities:
        city = city_data["city"]
        country = normalize_country(city_data["country"])
        key = (city.lower(), country.lower())
        if key not in seen:
            seen.add(key)
            records.append({
                "city": city,
                "country": country,
                "region": city_data["region"],
                "lat": city_data.get("lat"),
                "lon": city_data.get("lon"),
            })

    region_order = {"Africa": 0, "Americas": 1, "Asia": 2, "Europe": 3, "Oceania": 4}
    records.sort(key=lambda r: (region_order.get(r["region"], 99), r["country"], r["city"]))
    return records


# ---------------------------------------------------------------------------
# Data source 3: Wikidata enrichment (population, founded, lat/lon fallback)
# ---------------------------------------------------------------------------

def _request_json(url: str, params: dict | None = None, retries: int = 3) -> dict | None:
    """GET a URL and parse JSON, retrying on transient errors. Returns None on failure."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            if attempt == retries - 1:
                print(f"  ⚠ Request failed after {retries} attempts ({url}): {e}")
                return None
            time.sleep(2 ** attempt)  # exponential backoff: 1s, 2s, 4s
    return None


def _wikidata_search(city: str, country: str) -> str | None:
    """Search Wikidata for a city entity, returning the QID of the best match."""
    data = _request_json(
        WIKIDATA_SEARCH_URL,
        params={
            "action": "wbsearchentities",
            "search": city,
            "language": "en",
            "format": "json",
            "limit": 5,
            "type": "item",
        },
    )
    if data is None:
        return None
    results = data.get("search", [])

    if not results:
        return None

    # Prefer results whose description mentions "city", "capital", "town"
    city_keywords = ["city", "capital", "town", "metropolis", "municipality"]
    for result in results:
        desc = result.get("description", "").lower()
        if any(kw in desc for kw in city_keywords):
            return result["id"]

    # Fallback: return first result
    return results[0]["id"] if results else None


def _wikidata_entity(qid: str) -> dict:
    """Fetch entity data from Wikidata and extract population, founded, coords."""
    data = _request_json(WIKIDATA_ENTITY_URL.format(qid))
    if data is None or "entities" not in data or qid not in data["entities"]:
        return {"population": None, "founded": None, "lat": None, "lon": None}
    entity = data["entities"][qid]
    claims = entity.get("claims", {})

    result = {"population": None, "founded": None, "lat": None, "lon": None}

    # --- Population (P1082) — take the most recent value ---
    pop_claims = claims.get("P1082", [])
    best_pop = None
    best_year = 0
    for claim in pop_claims:
        try:
            v = claim["mainsnak"]["datavalue"]["value"]
            amount = v["amount"] if isinstance(v, dict) else v
            pop_val = int(float(str(amount).replace("+", "")))

            # Extract year from pointInTime qualifier (P585)
            year = 0
            for qual in claim.get("qualifiers", {}).get("P585", []):
                try:
                    time_val = qual["datavalue"]["value"]["time"]
                    year = int(time_val[1:5])
                except (KeyError, IndexError, ValueError):
                    pass

            if year >= best_year:
                best_year = year
                best_pop = pop_val
        except (KeyError, IndexError, ValueError, TypeError):
            continue

    result["population"] = best_pop

    # --- Founded / Inception (P571) ---
    for claim in claims.get("P571", []):
        try:
            v = claim["mainsnak"]["datavalue"]["value"]
            time_str = v.get("time", "")
            if time_str:
                is_bce = time_str.startswith("-")
                year_str = time_str[1:5] if is_bce else time_str[1:5]
                year_int = int(year_str)
                precision = v.get("precision", 9)
                if precision >= 9:  # year-level precision or better
                    result["founded"] = f"{year_int} BCE" if is_bce else str(year_int)
                elif precision == 8:  # decade-level
                    result["founded"] = f"{year_int}s BCE" if is_bce else f"{year_int}s"
                elif precision == 7:  # century-level
                    century = (year_int // 100) + 1
                    result["founded"] = f"{century}th c. BCE" if is_bce else f"{century}th c."
            break
        except (KeyError, IndexError, ValueError, TypeError):
            continue

    # --- Coordinates (P625) - as fallback if REST Countries didn't have them ---
    for claim in claims.get("P625", []):
        try:
            v = claim["mainsnak"]["datavalue"]["value"]
            result["lat"] = round(v["latitude"], 4)
            result["lon"] = round(v["longitude"], 4)
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        break

    return result


def enrich_from_wikidata(records: list[dict]) -> list[dict]:
    """Enrich city records with population, founded year, and coordinates
    from Wikidata. Modifies records in-place."""
    print(f"\nEnriching {len(records)} cities from Wikidata...")
    print("  (this takes ~2-3 minutes due to rate limiting)")

    enriched = 0
    for i, record in enumerate(records):
        if (i + 1) % 20 == 0:
            print(f"  Progress: {i + 1}/{len(records)} cities...")

        try:
            qid = _wikidata_search(record["city"], record["country"])
        except Exception as e:
            print(f"  ⚠ Error searching {record['city']}: {e}")
            time.sleep(2)
            continue

        if qid is None:
            continue

        time.sleep(WIKIDATA_DELAY)  # Rate limit

        try:
            data = _wikidata_entity(qid)
        except Exception as e:
            print(f"  ⚠ Error fetching {record['city']}: {e}")
            time.sleep(2)
            continue

        # Only overwrite if we got better data
        if data["population"] is not None:
            record["population"] = data["population"]
        if data["founded"] is not None:
            record["founded"] = data["founded"]
        if record.get("lat") is None and data["lat"] is not None:
            record["lat"] = data["lat"]
        if record.get("lon") is None and data["lon"] is not None:
            record["lon"] = data["lon"]

        enriched += 1
        time.sleep(WIKIDATA_DELAY)

    print(f"  Enriched {enriched}/{len(records)} cities")
    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("World Cities Data Fetcher")
    print("=" * 60)

    # Step 1: Fetch capitals from REST Countries
    capitals = fetch_countries()

    # Step 2: Major non-capital cities
    major_cities = fetch_major_non_capital_cities()

    # Step 3: Merge & deduplicate
    print("\nMerging and deduplicating...")
    records = merge_city_data(capitals, major_cities)
    print(f"Total cities: {len(records)}")

    # Step 4: Enrich with Wikidata (population, founded, lat/lon fallback)
    records = enrich_from_wikidata(records)

    # Step 5: Assign business activity classification
    print("\nAssigning business activity levels...")
    for record in records:
        record["business_activity"] = classify_business_activity(
            record["city"], record.get("population")
        )

    # Step 6: Write output
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "city", "country", "region", "lat", "lon",
        "population", "founded", "business_activity",
    ]

    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({
                k: record.get(k, "") for k in fieldnames
            })

    print(f"\n✅ Saved to {OUTPUT_PATH}")

    # Summary
    from collections import Counter
    print(f"\nTotal: {len(records)} cities")
    print("\nBy region:")
    for region, count in sorted(Counter(r["region"] for r in records).items()):
        print(f"  {region}: {count}")

    print("\nBy business activity level:")
    for level, count in sorted(
        Counter(r["business_activity"] for r in records).items(),
        key=lambda x: x[1], reverse=True,
    ):
        print(f"  {level}: {count}")

    print("\nSample rows:")
    for record in records[:10]:
        pop = record.get('population', 'N/A')
        pop_str = f"{pop:,}" if isinstance(pop, int) else str(pop)
        print(
            f"  {record['city']}, {record['country']}, {record['region']}"
            f" | pop={pop_str}"
            f" | founded={record.get('founded', 'N/A')}"
            f" | biz={record['business_activity']}"
        )


if __name__ == "__main__":
    main()