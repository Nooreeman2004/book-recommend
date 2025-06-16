Big Data Analytics - Book Recommendation System
Overview
This project implements a Book Recommendation System leveraging data analytics and machine learning techniques. It processes large datasets to recommend books to users based on their preferences and behaviors. The project showcases skills in data preprocessing, machine learning model development, and cloud deployment.

Features
Data Preprocessing: Handles large datasets, including cleaning and transformation.

Recommendation Engine: Combines collaborative filtering, content-based filtering, and hybrid approaches.

Frontend Integration: Interactive user interface built using Streamlit.

Backend API: RESTful API powered by FastAPI.

Cloud Integration: Supports large file storage using AWS S3.

Scalable Deployment: Dockerized setup hosted on AWS ECS.

Dataset
The project uses the following datasets:

Books.csv: Contains metadata about books, including titles, authors, and publication details.

Ratings.csv: User ratings data for books.

Processed Ratings: Preprocessed data for optimized recommendation performance.

Model Comparison: Results of various model evaluations.

Installation
Clone the repository:

bash
Copy
Edit
git clone https://github.com/Nooreeman2004/book-recommend.git
cd book-recommend
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Set up environment variables:

Create a .env file and include your AWS S3 credentials and other necessary configurations.

Run the backend:

bash
Copy
Edit
uvicorn main:app --reload
Run the frontend:

bash
Copy
Edit
streamlit run app.py
Project Structure
plaintext
Copy
Edit
project/
├── backend/
│   ├── models/                # Pre-trained and generated models
│   ├── main.py                # FastAPI backend code
│   └── requirements.txt       # Python dependencies for backend
├── frontend/
│   ├── app.py                 # Streamlit application
│   └── requirements.txt       # Python dependencies for frontend
├── data/
│   ├── Books.csv              # Book metadata
│   ├── Ratings.csv            # User ratings
│   ├── Processed_Ratings.csv  # Preprocessed ratings data
├── README.md                  # Project documentation
└── Dockerfile                 # Docker configuration
How to Use
Upload your data files (Books.csv and Ratings.csv) to the data/ directory.

Preprocess the data using the provided scripts.

Train the recommendation model and save it to backend/models/.

Start the backend and frontend services.

Access the application via your browser to get personalized book recommendations.

Cloud Integration
The project uses AWS S3 to store large datasets and models. Ensure your .env file contains the following:

env
Copy
Edit
AWS_ACCESS_KEY_ID=<your-access-key>
AWS_SECRET_ACCESS_KEY=<your-secret-key>
S3_BUCKET_NAME=<your-bucket-name>
Contributions
Contributions are welcome! Please open a pull request or an issue if you want to improve the project.

License
This project is licensed under the MIT License. See the LICENSE file for details.

