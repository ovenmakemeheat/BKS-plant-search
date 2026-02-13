# Plant Identify

## Installation

Install the project dependencies using uv:

```bash
pip install uv
uv sync
```

## Usage

### Development Server

Start the development server using uv:

```bash
uv run uvicorn src.main:app --reload
```

### Production Server

For production deployment, start the server with host and port configuration:

```bash
uv run uvicorn src.main:app --host 0.0.0.0 --port 8000
```

You can customize the host and port as needed for your deployment environment.

### API Usage

The application provides a FastAPI-based API for plant identification. After starting the server, you can access the API endpoints.

Example using uv:

```bash
uv run uvicorn src.main:app
```

This will start the FastAPI server with auto-reload enabled for development.

### API Documentation

Once the server is running, you can access the interactive API documentation at:

- **Swagger UI**: http://localhost:8000/docs

The Swagger UI provides an interactive interface where you can test API endpoints directly in your browser.

### Docker Deployment

Build and run the application using Docker:

```bash
# Build the Docker image
docker build -t plant-identify .

# Run the container
docker run -p 8000:8000 plant-identify
```

The application will be available at http://localhost:8000
