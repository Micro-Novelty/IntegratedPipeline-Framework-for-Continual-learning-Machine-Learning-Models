"""Singleton metaclass / base class used to guarantee a single shared instance."""

import threading
from typing import Any, Dict


class SingletonMeta(type):
    """Thread-safe singleton metaclass"""
    _instances: Dict[type, Any] = {}
    _lock: threading.Lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        # Fast path: instance already exists
        if cls in cls._instances:
            return cls._instances[cls]
        
        # Slow path: create instance with lock
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
            return cls._instances[cls]
    
    @classmethod
    def clear_instance(cls, target_class):
        """Clear singleton instance (useful for testing)"""
        with cls._lock:
            if target_class in cls._instances:
                del cls._instances[target_class]
    
    @classmethod
    def get_instance(cls, target_class):
        """Get instance without creating"""
        return cls._instances.get(target_class)


class Singleton(metaclass=SingletonMeta):
    """Base singleton class - inherit from this"""
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        # This is handled by metaclass, but for clarity
        return super().__new__(cls)
    
    def __init__(self, *args, **kwargs):
        if self._initialized:
            print(f"[===] Reusing existing {self.__class__.__name__} instance (id: {id(self)})")
            return
        self._initialized = True
        print(f"[===] Creating NEW {self.__class__.__name__} instance (id: {id(self)})")
