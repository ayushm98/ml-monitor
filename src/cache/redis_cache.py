"""Redis caching for prediction results."""
import json
from typing import Optional
import hashlib

class PredictionCache:
    """Cache for fraud predictions to reduce duplicate computations."""
    
    def __init__(self, ttl: int = 3600):
        """Initialize cache with time-to-live."""
        self.ttl = ttl
        self.cache = {}  # In-memory cache (replace with Redis in production)
    
    def get_cache_key(self, features: dict) -> str:
        """Generate cache key from features."""
        return hashlib.md5(json.dumps(features, sort_keys=True).encode()).hexdigest()
    
    def get(self, features: dict) -> Optional[dict]:
        """Get cached prediction."""
        key = self.get_cache_key(features)
        return self.cache.get(key)
    
    def set(self, features: dict, prediction: dict):
        """Cache prediction result."""
        key = self.get_cache_key(features)
        self.cache[key] = prediction

cache = PredictionCache()
