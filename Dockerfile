# Use slim python based image
FROM python:3.11-slim

# Set working directory inside container
WORKDIR /app

# Copy dependencies files first (better caching)
COPY pyproject.toml uv.lock* ./

# Install uv
RUN pip install uv
RUN uv sync --frozen --no-dev

# Copy the project files 
COPY . .

# Expose FastAPI default port
EXPOSE 8000

# Command to run app with uvicorn
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]