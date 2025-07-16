import redis
import pickle
from src.config.settings import settings

class CacheManager:
    def __init__(self):
        self.client = redis.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            db=settings.REDIS_DB,
            decode_responses=False
        )

    def get(self, key):
        value = self.client.get(key)
        if value:
            return pickle.loads(value)
        return None

    def set(self, key, value, ttl=60):
        self.client.setex(key, ttl, pickle.dumps(value)) 