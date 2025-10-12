# api/routes/__init__.py
from .nlp import router as nlp_router
from .prediction import router as prediction_router
from .chat import router as chat_router
from .analysis import router as analysis_router
from .admin import router as admin_router
from .health import router as health_router

__all__ = [
    "nlp_router",
    "prediction_router",
    "chat_router",
    "analysis_router",
    "admin_router",
    "health_router"
]