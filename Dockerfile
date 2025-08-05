FROM python:3.13-slim

WORKDIR /app

# Install setuptools first to avoid build issues
RUN pip install --no-cache-dir setuptools wheel

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port for Streamlit
EXPOSE 8501

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Command to run the unified app (replacing the previous command)
CMD ["streamlit", "run", "app/unified_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
