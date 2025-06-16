from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import load_npz
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import Optional

app = FastAPI(title="Book Recommendation API")

# Define paths
MODEL_DIR = "D:/data/6th sem/Big data analytics/theory project/archive/backend/models"
DATA_DIR = "D:/data/6th sem/Big data analytics/theory project/archive/data"
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, "processed_ratings_with_details.csv")
TFIDF_MATRIX_PATH = os.path.join(MODEL_DIR, "tfidf_matrix.npz")
TFIDF_VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "XGBoost.joblib")
MODEL_COMPARISON_PATH = os.path.join(DATA_DIR, "model_comparison.csv")
BOOKS_CSV_PATH = os.path.join(DATA_DIR, "Books.csv")

# Load assets with validation
try:
    ratings_with_details = pd.read_csv(PROCESSED_DATA_PATH)
    books_df = pd.read_csv(BOOKS_CSV_PATH, encoding='latin1', dtype={'Year-Of-Publication': str})
    tfidf_matrix = load_npz(TFIDF_MATRIX_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)
    xgb_model = joblib.load(XGB_MODEL_PATH)
    # Validate tfidf_matrix and books_df alignment
    if tfidf_matrix.shape[0] != books_df.shape[0]:
        raise ValueError(f"tfidf_matrix rows ({tfidf_matrix.shape[0]}) do not match books_df rows ({books_df.shape[0]})")
except FileNotFoundError as e:
    raise Exception(f"Failed to load required file: {e}")
except Exception as e:
    raise Exception(f"Initialization error: {str(e)}")

# Define request model
class RecommendationRequest(BaseModel):
    user_id: int
    book_title: str

# Recommendation function
def recommend_book(user_id: int, book_title: str):
    feature_columns = ['Age', 'Avg_Book_Rating', 'Avg_User_Rating', 'Decade', 'Country_Encoded']
    
    # Collaborative filtering: Predict high rating probability
    user_data = ratings_with_details[ratings_with_details['User-ID'] == user_id][feature_columns].mean()
    if user_data.empty or user_data.isna().any():
        user_data = ratings_with_details[feature_columns].mean()
    pred_proba = xgb_model.predict_proba([user_data])[0][1] if hasattr(xgb_model, 'predict_proba') else 0.5
    
    # Content-based filtering: Find similar books
    book_idx = books_df[books_df['Book-Title'].str.contains(book_title, case=False, na=False)].index
    if not book_idx.empty:
        book_idx = book_idx[0]
        similarities = cosine_similarity(tfidf_matrix[book_idx:book_idx+1], tfidf_matrix).flatten()
        similar_indices = similarities.argsort()[-10:][::-1]
        # Validate indices
        valid_indices = similar_indices[similar_indices < len(books_df)]
        if len(valid_indices) == 0:
            raise ValueError("No valid similar book indices found")
        similar_books = books_df.iloc[valid_indices]
        book_avg_rating = ratings_with_details.groupby('ISBN')['Book-Rating'].mean().reset_index(name='Avg_Book_Rating')
        similar_books = similar_books.merge(book_avg_rating, on='ISBN', how='left')
        similar_books = similar_books[similar_books['Avg_Book_Rating'].notna() & (similar_books['Avg_Book_Rating'] >= 7)]
        if not similar_books.empty:
            return {
                "book_title": similar_books.iloc[0]['Book-Title'],
                "author": similar_books.iloc[0]['Book-Author'],
                "avg_rating": float(similar_books.iloc[0]['Avg_Book_Rating']),
                "probability": float(pred_proba)
            }
    
    # Default to most popular book
    popular_book = ratings_with_details.merge(books_df[['ISBN', 'Book-Title', 'Book-Author']], on='ISBN')\
                                      .groupby(['ISBN', 'Book-Title', 'Book-Author'])['Book-Rating'].mean()\
                                      .reset_index().sort_values('Book-Rating', ascending=False).iloc[0]
    return {
        "book_title": popular_book['Book-Title'],
        "author": popular_book['Book-Author'],
        "avg_rating": float(popular_book['Book-Rating']),
        "probability": float(pred_proba)
    }

# Endpoints
@app.get("/")
async def root():
    return {"message": "Book Recommendation API"}

@app.post("/recommend")
async def get_recommendation(request: RecommendationRequest):
    try:
        recommendation = recommend_book(request.user_id, request.book_title)
        return {
            "status": "success",
            "recommended_book": recommendation["book_title"],
            "author": recommendation["author"],
            "average_rating": recommendation["avg_rating"],
            "high_rating_probability": recommendation["probability"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendation: {str(e)}")

@app.get("/model-performance")
async def get_model_performance():
    try:
        # Load the CSV with the index column (model names) as the index
        results_df = pd.read_csv(MODEL_COMPARISON_PATH, index_col=0)
        # Convert to dictionary with model names as keys
        metrics = results_df.to_dict(orient='index')
        return {
            "status": "success",
            "models": metrics
        }
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Model comparison file not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model performance: {str(e)}")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)