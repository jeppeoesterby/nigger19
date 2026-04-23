"""WSGI entry point for gunicorn/any production WSGI server."""
from webapp.app import create_app

app = create_app()
