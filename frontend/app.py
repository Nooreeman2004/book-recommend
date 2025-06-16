import streamlit as st
import requests
# Streamlit app configuration
st.set_page_config(page_title="Book Recommendation System", layout="centered")

# Backend API URL
API_URL = "http://backend/recommend"  # No trailing slash

# App Title with Logo
st.markdown(
    """
    <div style="display: flex; align-items: center; justify-content: center;">
        <img src="https://img.icons8.com/color/96/000000/books.png" alt="Books Logo" height="80" style="margin-right: 10px;"/>
        <h1 style="display: inline;">Book Recommendation System</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Input Section
st.write("### Enter Details to Get a Book Recommendation")

# Input fields
user_id = st.number_input("User ID", min_value=1, step=1, value=276725, help="Enter your unique User ID (e.g., 276725)")
book_title = st.text_input("Book Title", value="Clara Callan", help="Enter the title of a book you like (e.g., Clara Callan)")

# Recommendation Button
if st.button("Get Recommendation"):
    if user_id and book_title:
        payload = {
            "user_id": int(user_id),
            "book_title": book_title.strip()
        }
        try:
            # Call the backend API
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                result = response.json()
                # Display recommendation
                st.write("### Recommended Book")
                st.success(f"**Title:** {result['recommended_book']}")
                st.write(f"**Author:** {result['author']}")
                st.write(f"**Average Rating:** {result['average_rating']:.2f}")
                st.write(f"**High Rating Probability:** {result['high_rating_probability']:.2%}")
            else:
                st.error(f"Error: {response.json().get('detail', 'Failed to get recommendation')}")
        except requests.ConnectionError:
            st.error("Cannot connect to the backend. Please ensure the API is running at http://127.0.0.1:8000.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please provide both User ID and Book Title.")

# Model Performance Section
st.write("### Model Performance")
if st.button("View Model Metrics"):
    try:
        response = requests.get("http://backend:8000/model-performance")
        if response.status_code == 200:
            metrics = response.json()['models']
            for model, data in metrics.items():
                st.write(f"**{model}**")
                st.write(f"- Accuracy: {data['Accuracy']:.4f}")
                st.write(f"- Log Loss: {data['Log-Loss']:.4f}")
                st.write(f"- ROC AUC: {data['ROC-AUC']:.4f}")
        else:
            st.error(f"Error: {response.json().get('detail', 'Failed to fetch metrics')}")
    except requests.ConnectionError:
        st.error("Cannot connect to the backend for metrics.")
    except Exception as e:
        st.error(f"An error occurred while fetching metrics: {str(e)}")

# Footer
st.markdown("---")
st.markdown("**Developed by Noor | Big Data Analytics Project**")