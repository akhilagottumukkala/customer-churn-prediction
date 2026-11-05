# Start from a small Linux image that already has Python 3.13
FROM python:3.13-slim

# Do not buffer logs, so they show up immediately
ENV PYTHONUNBUFFERED=1

# All following commands run inside /app in the container
WORKDIR /app

# Install libraries first, so Docker can reuse this step when only code changes
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy only what the API needs at runtime
COPY src/ src/
COPY models/ models/

# The API listens on port 8000
EXPOSE 8000

# Start the server. 0.0.0.0 lets traffic in from outside the container.
CMD ["uvicorn", "api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
