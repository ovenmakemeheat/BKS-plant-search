import io
import logging
import os

import numpy as np
import timm
import torch
import uvicorn
from depth_anything_3.api import DepthAnything3
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download
from PIL import Image
from pydantic import BaseModel

from src.model.CNN_regression import DBHRegressor, get_eval_transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
MODEL_CONFIG = {
    "model_name": "microhum/resnet50_224_depth_dbh_DA3_LARGE_20260210_0423_38_best",
    "model_file": "resnet50_224_depth_dbh_DA3_LARGE_20260210_0423_38_best.pt",
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}


# Load the model
def load_model():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_CONFIG["model_name"], filename=MODEL_CONFIG["model_file"]
        )

        # Create model architecture using DBHRegressor from CNN_regression
        model = DBHRegressor("resnet50")

        # Load trained weights
        state_dict = torch.load(model_path, map_location=MODEL_CONFIG["device"])

        # Data Parallel Handler
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        model.load_state_dict(state_dict)

        model = model.to(torch.device(MODEL_CONFIG["device"]))

        model.eval()

        logger.info(f"Model loaded successfully on {MODEL_CONFIG['device']}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise


# Preprocessing transforms
def preprocess_image(image: Image.Image, image_size: int = 224) -> torch.Tensor:
    """Preprocess image for model inference using transforms from CNN_regression"""
    # Convert to grayscale if needed
    if image.mode != "L":
        image = image.convert("L")

    # Use the evaluation transforms from CNN_regression
    transform = get_eval_transforms(image_size, in_channels=1)

    # Apply transforms
    image = transform(image)

    return image


# Initialize models
model = load_model()


# Initialize Depth Anything v3 for depth estimation
def load_depth_model(model_name="depth-anything/DA3NESTED-GIANT-LARGE-1.1"):
    """Load the Depth Anything v3 model"""
    try:
        depth_model = DepthAnything3.from_pretrained(model_name)
        depth_model = depth_model.to(MODEL_CONFIG["device"])
        depth_model.eval()
        logger.info("Depth Anything v3 model loaded successfully")
        return depth_model
    except Exception as e:
        logger.error(f"Error loading Depth Anything v3 model: {e}")
        raise


depth_model = load_depth_model("depth-anything/DA3-LARGE-1.1")

# FastAPI app
app = FastAPI(
    title="Tree DBH Estimation API",
    description="API for estimating tree Diameter at Breast Height (DBH) from depth maps",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionResponse(BaseModel):
    dbh: float
    confidence: float
    raw_prediction: float


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Tree DBH Estimation API"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "device": MODEL_CONFIG["device"]}


@app.post("/predict", response_model=PredictionResponse)
async def predict_dbh(file: UploadFile = File(...)):
    """
    Predict tree DBH from a depth map image.

    Args:
        file: Depth map image file (PNG, JPG, etc.)

    Returns:
        PredictionResponse: Contains DBH estimation and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload an image file."
            )

        # Read and validate image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Preprocess image
        input_tensor = preprocess_image(image)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(MODEL_CONFIG["device"])

        # Make prediction
        with torch.no_grad():
            raw_prediction = model(input_tensor).squeeze().item()

        # Convert from log scale back to original scale
        dbh_prediction = np.expm1(raw_prediction)

        # Calculate confidence (simple heuristic based on prediction magnitude)
        # This is a placeholder - in practice, you might want to use model uncertainty
        confidence = min(1.0, max(0.0, 1.0 - abs(raw_prediction) * 0.1))

        print(dbh_prediction, confidence, raw_prediction)

        return PredictionResponse(
            dbh=round(dbh_prediction, 2),
            confidence=round(confidence, 3),
            raw_prediction=round(raw_prediction, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_from_rgb", response_model=PredictionResponse)
async def predict_dbh_from_rgb(file: UploadFile = File(...)):
    """
    Estimate depth from RGB image and predict tree DBH.

    Args:
        file: RGB image file (PNG, JPG, etc.)

    Returns:
        PredictionResponse: Contains DBH estimation and confidence
    """
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload an image file."
            )

        # Read and validate image
        image_bytes = await file.read()
        if len(image_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Open image
        image = Image.open(io.BytesIO(image_bytes))

        # Estimate depth using Depth Anything v3
        prediction = depth_model.inference(
            [image],
            use_ray_pose=True,
            ref_view_strategy="saddle_balanced",
            export_feat_layers=[0, 1, 2],
        )

        print(prediction)

        # Get depth map (N, H, W) - take first image
        depth_map = prediction.depth[0].astype(np.float32)  # Shape: (H, W)
        depth_map = np.clip(depth_map, 0, 255).astype(np.uint8)
        depth_map = Image.fromarray(depth_map, mode="L")

        # Preprocess depth image for DBH model
        input_tensor = preprocess_image(depth_map)
        input_tensor = input_tensor.unsqueeze(0)
        input_tensor = input_tensor.to(MODEL_CONFIG["device"])

        # Make prediction
        with torch.no_grad():
            raw_prediction = model(input_tensor).squeeze().item()

        # Convert from log scale back to original scale
        dbh_prediction = np.expm1(raw_prediction)

        # Calculate confidence (simple heuristic based on prediction magnitude)
        confidence = min(1.0, max(0.0, 1.0 - abs(raw_prediction) * 0.1))

        return PredictionResponse(
            dbh=round(dbh_prediction, 2),
            confidence=round(confidence, 3),
            raw_prediction=round(raw_prediction, 4),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/batch_predict")
async def batch_predict(files: list[UploadFile] = File(...)):
    """
    Predict DBH for multiple images.

    Args:
        files: List of depth map image files

    Returns:
        List of predictions for each image
    """
    if len(files) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum 100 files per batch.")

    results = []

    for file in files:
        try:
            # Same processing as single prediction
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))
            input_tensor = preprocess_image(image)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(MODEL_CONFIG["device"])

            with torch.no_grad():
                raw_prediction = model(input_tensor).squeeze().item()

            dbh_prediction = np.expm1(raw_prediction)
            confidence = min(1.0, max(0.0, 1.0 - abs(raw_prediction) * 0.1))

            results.append(
                {
                    "filename": file.filename,
                    "dbh": round(dbh_prediction, 2),
                    "confidence": round(confidence, 3),
                    "raw_prediction": round(raw_prediction, 4),
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results}


@app.post("/batch_predict_from_rgb")
async def batch_predict_from_rgb(files: list[UploadFile] = File(...)):
    """
    Estimate depth from RGB images and predict DBH for multiple images.

    Args:
        files: List of RGB image files

    Returns:
        List of predictions for each image
    """
    if len(files) > 100:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files. Maximum 100 files per batch.")

    results = []

    for file in files:
        try:
            # Read and validate image
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Estimate depth using Depth Anything v3
            prediction = depth_model.inference(
                [image],
                use_ray_pose=True,
                ref_view_strategy="saddle_balanced",
                export_feat_layers=[0, 1, 2],
            )

            # Get depth map and convert to PIL Image
            depth_map = prediction.depth[0]
            # depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min())
            # depth_uint8 = (depth_normalized * 255).astype(np.uint8)
            depth_image = Image.fromarray(depth_map, mode="L")

            # Preprocess depth image for DBH model
            input_tensor = preprocess_image(depth_image)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(MODEL_CONFIG["device"])

            # Make prediction
            with torch.no_grad():
                raw_prediction = model(input_tensor).squeeze().item()

            dbh_prediction = np.expm1(raw_prediction)
            confidence = min(1.0, max(0.0, 1.0 - abs(raw_prediction) * 0.1))

            results.append(
                {
                    "filename": file.filename,
                    "dbh": round(dbh_prediction, 2),
                    "confidence": round(confidence, 3),
                    "raw_prediction": round(raw_prediction, 4),
                }
            )

        except Exception as e:
            results.append({"filename": file.filename, "error": str(e)})

    return {"predictions": results}


def main():
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(description="Agentic RAG API and Flows")
    parser.add_argument("command", choices=["serve"], help="Command to run.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for the API server.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    parser.add_argument(
        "--reload",
        default=True,
        action="store_true",
        help="Enable auto-reloading for development.",
    )

    args = parser.parse_args()

    if args.command == "serve":
        print(f"Starting API server on {args.host}:{args.port}")

        uvicorn.run("api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
