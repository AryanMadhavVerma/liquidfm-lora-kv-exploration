FROM python:3.11-slim

WORKDIR /app

# install dependencies first (layer cached unless requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy server code
COPY server/ server/

# copy merged model weights
# this makes the image self-contained (~2.5GB larger)
COPY models/eli5-merged/ models/eli5-merged/

# uvicorn serves on 8000 inside the container
EXPOSE 8000

# CPU-only inside Docker on Mac (no GPU passthrough)
ENV DEVICE=cpu

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
