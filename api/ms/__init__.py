# Imports
from fastapi import FastAPI
import joblib

# Initialize FastAPI app
app = FastAPI()

# Load model
model = joblib.load('model/random_forest.pickle')
