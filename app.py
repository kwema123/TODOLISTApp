import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime

# Database configuration
DB_CONFIG = {
    'host': 'localhost',
    'database': 'student_performance',
    'user': 'root',
    'password': 'K@minu12',
    'raise_on_warnings': True
}

# Set page config
st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide",
    page_icon="🎓"
)

# Initialize database connection
def get_db_connection():
    try:
        connection = mysql.connector.connect(**DB_CONFIG)
        return connection
    except Error as e:
        st.error(f"Database connection failed: {e}")
        return None

# Enhanced database initialization
def init_database():
    connection = None
    try:
        connection = get_db_connection()
        if connection is None:
            return False
            
        cursor = connection.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            attendance DECIMAL(5,2),
            assignment_score DECIMAL(5,2),
            midterm_score DECIMAL(5,2),
            final_score DECIMAL(5,2),
            outstanding_balance INT,
            prediction VARCHAR(10),
            confidence DECIMAL(5,2),
            prediction_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        connection.commit()
        return True
        
    except Error as e:
        if e.errno == 1050:
            st.warning("Tables already exist - using existing structure")
            return True
        st.error(f"Database initialization error: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Store prediction with verification
def store_prediction(data, prediction, confidence):
    connection = None
    try:
        connection = get_db_connection()
        if connection is None:
            return False
            
        cursor = connection.cursor()
        
        insert_query = """
        INSERT INTO predictions 
        (attendance, assignment_score, midterm_score, final_score, outstanding_balance, prediction, confidence)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        record = (
            float(data['Attendance']),
            float(data['Assignment_Score']),
            float(data['Midterm_Score']),
            float(data['Final_Score']),
            int(data['Outstanding_Balance']),
            str(prediction),
            float(confidence)
        )
        
        cursor.execute(insert_query, record)
        connection.commit()
        
        st.success("✅ Prediction saved successfully!")
        return True
        
    except Error as e:
        st.error(f"❌ Failed to save prediction: {e}")
        return False
    finally:
        if connection and connection.is_connected():
            cursor.close()
            connection.close()

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_path = 'models/retrained_rf_model.pkl'
        features_path = 'models/retrained_rf_features.pkl'
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found at {features_path}")
            
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        return model, features
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Main dashboard function
def main():
    st.title("🎓 Student Performance Predictor")
    
    # Initialize database and load model
    if init_database():
        st.success("Database ready")
    else:
        st.error("Database initialization failed")

    model, features = load_model()

    # Database management in sidebar
    with st.sidebar:
        st.header("⚙️ Database Tools")
        
        if st.button("🔄 Check Connection"):
            conn = get_db_connection()
            if conn:
                st.success("✅ Connected")
                conn.close()
            else:
                st.error("❌ Connection failed")
        
        if st.button("🧹 Clear Predictions", type="secondary"):
            connection = get_db_connection()
            if connection:
                try:
                    cursor = connection.cursor()
                    cursor.execute("TRUNCATE TABLE predictions")
                    connection.commit()
                    st.success("Predictions cleared")
                except Error as e:
                    st.error(f"Error: {e}")
                finally:
                    if connection.is_connected():
                        connection.close()
        
        st.divider()
        st.header("📊 Input Data")
        
        inputs = {
            'Attendance': st.slider("Attendance (%)", 0, 100, 80),
            'Assignment_Score': st.slider("Assignment Score", 0, 100, 70),
            'Midterm_Score': st.slider("Midterm Score", 0, 100, 65),
            'Final_Score': st.slider("Final Score", 0, 100, 60),
            'Outstanding_Balance': st.slider("Outstanding Balance ($)", 0, 5000, 1000)
        }

    if model and features:
        if st.button("🚀 Predict Performance", type="primary"):
            with st.spinner('Analyzing...'):
                # Prepare input data
                input_df = pd.DataFrame([inputs], columns=features)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                proba = model.predict_proba(input_df)[0]
                confidence = np.max(proba) * 100

                # Display results
                st.header("📋 Results")
                col1, col2 = st.columns(2)
                with col1:
                    emoji = "✅" if prediction == "Pass" else "❌"
                    st.metric("Prediction", f"{emoji} {prediction}")
                with col2:
                    st.metric("Confidence", f"{confidence:.1f}%")

                # Save to database (balloon animation removed)
                store_prediction(inputs, prediction, confidence)
                
                # Model insights
                st.header("📊 Model Insights")
                
                tab1, tab2 = st.tabs(["Feature Importance", "Decision Tree"])
                
                with tab1:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    importance = pd.DataFrame({
                        'Feature': features,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    sns.barplot(data=importance, x='Importance', y='Feature', palette='viridis', ax=ax)
                    ax.set_title('Feature Importance')
                    st.pyplot(fig)
                
                with tab2:
                    fig, ax = plt.subplots(figsize=(20, 12))
                    plot_tree(model.estimators_[0], 
                             feature_names=features,
                             class_names=model.classes_,
                             filled=True,
                             ax=ax)
                    st.pyplot(fig)

                # Recent predictions
                st.header("📋 Recent Predictions")
                connection = get_db_connection()
                if connection:
                    try:
                        df = pd.read_sql(
                            """SELECT 
                                attendance, assignment_score, midterm_score, 
                                final_score, outstanding_balance, prediction,
                                confidence, 
                                DATE_FORMAT(prediction_time, '%Y-%m-%d %H:%i') as time
                            FROM predictions 
                            ORDER BY prediction_time DESC 
                            LIMIT 5""", 
                            connection
                        )
                        st.dataframe(df)
                    except Error as e:
                        st.error(f"Database error: {e}")
                    finally:
                        connection.close()

    else:
        st.error("System not ready. Please check:")
        st.error("1. Model files exist in 'models/' directory")
        st.error("2. Database is running and accessible")

if __name__ == "__main__":
    main()