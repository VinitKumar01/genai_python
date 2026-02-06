from .server import app
import uvicorn


def main():
    uvicorn.run(app=app, port=3000, host="0.0.0.0")


# python -m rag_queue.main to run the server
main()
