"""Gradio interface startup script (using FastAPI backend)."""

from app.api.gradio_app import create_gradio_app

if __name__ == "__main__":
    app = create_gradio_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True
    )
