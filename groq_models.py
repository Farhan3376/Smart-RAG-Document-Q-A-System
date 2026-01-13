import requests
import os
from typing import List, Dict, Optional
from functools import lru_cache
import json
from pathlib import Path

# Groq API endpoint
GROQ_MODELS_ENDPOINT = "https://api.groq.com/openai/v1/models"

# Fallback models (hardcoded defaults if API is unavailable)
FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma-7b-it"
]

# Models to exclude (speech-to-text, content moderation, etc.)
EXCLUDE_MODELS = {
    "whisper",           # Speech-to-text models
    "llama-guard",       # Content moderation
    "llama-prompt-guard", # Prompt safety
    "safeguard",         # Safety models
    "playai-tts",        # Text-to-speech
    "deprecated",        # Deprecated models
}


def fetch_groq_models() -> Optional[List[Dict]]:
    """
    Fetch all available models from Groq API.
    
    Returns:
        List of model dictionaries with 'id' and other metadata, or None if failed
    """
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️  GROQ_API_KEY not set, using fallback models")
            return None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(GROQ_MODELS_ENDPOINT, headers=headers, timeout=5)
        response.raise_for_status()
        
        data = response.json()
        return data.get("data", [])
    
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Failed to fetch Groq models: {str(e)}")
        return None
    except Exception as e:
        print(f"⚠️  Error processing Groq models: {str(e)}")
        return None


def is_text_model(model_id: str) -> bool:
    """
    Check if model is suitable for text generation (not speech/safety/etc).
    
    Args:
        model_id: The model ID to check
    
    Returns:
        True if model is a text generation model, False otherwise
    """
    model_id_lower = model_id.lower()
    
    # Exclude non-text models
    for exclude in EXCLUDE_MODELS:
        if exclude in model_id_lower:
            return False
    
    return True


def get_production_models(models: List[Dict]) -> List[str]:
    """
    Extract all available text generation models from API response.
    
    Args:
        models: List of model dictionaries from Groq API
    
    Returns:
        List of model IDs suitable for text generation
    """
    available_models = []
    
    for model in models:
        model_id = model.get("id", "")
        
        # Only include text generation models (exclude safety, speech, etc.)
        if is_text_model(model_id):
            available_models.append(model_id)
    
    return available_models


def get_preview_models(models: List[Dict]) -> List[str]:
    """
    Extract preview models from API response.
    
    Args:
        models: List of model dictionaries from Groq API
    
    Returns:
        List of preview model IDs
    """
    preview_models = []
    
    for model in models:
        model_id = model.get("id", "")
        
        # Check if explicitly marked as preview
        is_preview = "preview" in model_id.lower() and any(
            model.get(key) == "preview" 
            for key in ["status", "state", "type"]
        )
        
        # Only include text generation preview models
        if is_preview and is_text_model(model_id):
            preview_models.append(model_id)
    
    return preview_models


def get_available_models(
    include_preview: bool = False,
    use_cache: bool = True
) -> List[str]:
    """
    Get list of available Groq models for text generation.
    
    Args:
        include_preview: Include preview models in results (default: False)
        use_cache: Use cached models if available
    
    Returns:
        List of model IDs, sorted by quality
    """
    # Try to fetch from API
    models = fetch_groq_models()
    
    if models is None:
        print("✓ Using fallback Groq models")
        return FALLBACK_MODELS
    
    # Get all available models
    available = get_production_models(models)
    
    # Get preview models if requested
    preview = get_preview_models(models) if include_preview else []
    
    # Combine and return
    all_models = available + preview
    
    if not all_models:
        print("✓ No models found, using fallback models")
        return FALLBACK_MODELS
    
    print(f"✓ Fetched {len(available)} available models")
    if preview:
        print(f"✓ Fetched {len(preview)} preview models")
    
    return all_models


def get_model_info(model_id: str, models: Optional[List[Dict]] = None) -> Optional[Dict]:
    """
    Get information about a specific model.
    
    Args:
        model_id: The model ID to get info for
        models: Pre-fetched models list (optional)
    
    Returns:
        Model information dictionary or None if not found
    """
    if models is None:
        models = fetch_groq_models()
    
    if models is None:
        return None
    
    for model in models:
        if model.get("id") == model_id:
            return model
    
    return None


@lru_cache(maxsize=1)
def get_cached_models(include_preview: bool = False) -> List[str]:
    """
    Get cached list of models with memoization.
    
    Args:
        include_preview: Include preview models
    
    Returns:
        List of model IDs
    """
    return get_available_models(include_preview=include_preview)


def save_models_cache(models: List[str], cache_file: str = "models_cache.json"):
    """
    Save models list to local cache file.
    
    Args:
        models: List of model IDs to cache
        cache_file: Path to cache file
    """
    try:
        with open(cache_file, "w") as f:
            json.dump({"models": models, "cached_at": str(Path.cwd())}, f)
        print(f"✓ Cached models to {cache_file}")
    except Exception as e:
        print(f"⚠️  Failed to cache models: {str(e)}")


def load_models_cache(cache_file: str = "models_cache.json") -> Optional[List[str]]:
    """
    Load models list from local cache file.
    
    Args:
        cache_file: Path to cache file
    
    Returns:
        List of model IDs from cache, or None if file doesn't exist
    """
    try:
        if Path(cache_file).exists():
            with open(cache_file, "r") as f:
                data = json.load(f)
                return data.get("models", None)
    except Exception as e:
        print(f"⚠️  Failed to load cache: {str(e)}")
    
    return None


def get_recommended_models() -> Dict[str, List[str]]:
    """
    Get recommended models organized by use case.
    
    Returns:
        Dictionary with models organized by category
    """
    models = get_available_models()
    
    return {
        "fastest": [m for m in models if "8b" in m][:2],
        "most_powerful": [m for m in models if "70b" in m][:2],
        "balanced": [m for m in models if "mixtral" in m or "gemma" in m][:2],
        "all": models
    }


# Example usage
if __name__ == "__main__":
    print("🔄 Fetching Groq models...\n")
    
    # Get available models
    models = get_available_models(include_preview=False)
    print(f"\n📊 Available models ({len(models)}):")
    for i, model in enumerate(models, 1):
        print(f"   {i}. {model}")
    
    # Get recommendations
    print("\n🎯 Recommended models by use case:")
    recommendations = get_recommended_models()
    for category, model_list in recommendations.items():
        if category != "all" and model_list:
            print(f"   {category.upper()}: {model_list[0]}")