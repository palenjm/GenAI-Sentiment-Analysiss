# import packages
import streamlit as st
import pandas as pd
import os
import plotly.express as px
import requests
# import packages
import os
import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from dotenv import load_dotenv

# Load environment variables and verify
load_dotenv()

# Check if environment variables are loaded
if not os.getenv("HUGGINGFACE_API_KEY"):
    st.error("⚠️ HUGGINGFACE_API_KEY not found in environment variables!")
    st.info("Please make sure your .env file contains: HUGGINGFACE_API_KEY=your_token_here")


# ...existing code...
def get_dataset_path():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "data", "customer_reviews.csv")
    return csv_path
# ...existing code...

# Function to get sentiment using Hugging Face API
@st.cache_data
def get_sentiment(text):
    if not text or pd.isna(text):
        return "Neutral"
    try:
        # Use a sentiment model specifically designed for review analysis
        api_url = "https://api-inference.huggingface.co/models/Seethal/sentiment_analysis_generic_dataset"
        api_key = os.getenv('HUGGINGFACE_API_KEY')
        
        headers = {
            "Authorization": f"Bearer {api_key}"
        }
        payload = {"inputs": text}
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Check if the response is successful
        if response.status_code != 200:
            st.error("❌ Error analyzing sentiment. Please try again later.")
            return "Neutral"
            
        result = response.json()
        
        # Parse the result
        if isinstance(result, list) and len(result) > 0:
            # Get the first prediction
            prediction = result[0]
            if isinstance(prediction, list):
                # Sort predictions by score and get the highest
                sorted_preds = sorted(prediction, key=lambda x: x['score'], reverse=True)
                if sorted_preds:
                    label = sorted_preds[0]['label'].upper()
                    confidence = sorted_preds[0]['score']
                    
                    # Map the sentiment with confidence threshold
                    if confidence > 0.7:  # Only use high confidence predictions
                        if 'POSITIVE' in label or 'POS' in label:
                            return "Positive"
                        elif 'NEGATIVE' in label or 'NEG' in label:
                            return "Negative"
                    
            # If we can't get a high confidence prediction, look for positive/negative words
            text_lower = text.lower()
            positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'perfect', 'best', 'wonderful', 'fantastic']
            negative_words = ['bad', 'poor', 'terrible', 'horrible', 'waste', 'disappointed', 'worst', 'awful']
            
            # Count positive and negative words
            pos_count = sum(1 for word in positive_words if word in text_lower)
            neg_count = sum(1 for word in negative_words if word in text_lower)
            
            if pos_count > neg_count:
                return "Positive"
            elif neg_count > pos_count:
                return "Negative"
        
        return "Neutral"  # Default case if we can't determine sentiment
                
    except Exception as e:
        st.error(f"API error: {e}")
        return "Neutral"
            
        return "Neutral"  # Default case
    except Exception as e:
        st.error(f"API error: {e}")
        st.error(f"Response content: {response.content if 'response' in locals() else 'No response'}")
        return "Neutral"

st.title("🔍 GenAI Sentiment Analysis")
st.write("This is your GenAI-powered data processing app.")

# Layout two buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("📥 Load Dataset"):
        try:
            csv_path = get_dataset_path()
            df = pd.read_csv(csv_path)
            st.session_state["df"] = df.head(10)
            st.success("Dataset loaded successfully!")
        except FileNotFoundError:
            st.error("Dataset not found. Please check the file path.")

with col2:
    if st.button("🔍 Analyze Sentiment"):
        if "df" in st.session_state:
            try:
                with st.spinner("Analyzing sentiment..."):
                    st.session_state["df"].loc[:, "Sentiment"] = st.session_state["df"]["SUMMARY"].apply(get_sentiment)
                    st.success("Sentiment analysis completed!")
            except Exception as e:
                st.error(f"Something went wrong: {e}")
        else:
            st.warning("Please ingest the dataset first.")

# Display the dataset if it exists
if "df" in st.session_state:
    # Product filter dropdown
    st.subheader("🔍 Filter by Product")
    product = st.selectbox("Choose a product", ["All Products"] + list(st.session_state["df"]["PRODUCT"].unique()))
    st.subheader(f"📁 Reviews for {product}")

    if product != "All Products":
        filtered_df = st.session_state["df"][st.session_state["df"]["PRODUCT"] == product]
    else:
        filtered_df = st.session_state["df"]
    st.dataframe(filtered_df)

    # Visualization using Plotly if sentiment analysis has been performed
    if "Sentiment" in st.session_state["df"].columns:
        st.subheader(f"📊 Sentiment Breakdown for {product}")
        
        # # Calculate total counts
        # total_counts = st.session_state["df"]["Sentiment"].value_counts()
        # st.write("📈 Overall Statistics:")
        # col1, col2, col3, col4 = st.columns(4)
        # with col1:
        #     st.metric("Total Reviews", len(st.session_state["df"]))
        # with col2:
        #     st.metric("Positive", int(total_counts.get("Positive", 0)))
        # with col3:
        #     st.metric("Neutral", int(total_counts.get("Neutral", 0)))
        # with col4:
        #     st.metric("Negative", int(total_counts.get("Negative", 0)))
        
        sentiment_counts = filtered_df["Sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        sentiment_order = ['Negative', 'Neutral', 'Positive']
        sentiment_colors = {'Negative': 'red', 'Neutral': 'lightgray', 'Positive': 'green'}
        
        existing_sentiments = sentiment_counts['Sentiment'].unique()
        filtered_order = [s for s in sentiment_order if s in existing_sentiments]
        filtered_colors = {s: sentiment_colors[s] for s in existing_sentiments if s in sentiment_colors}
        
        sentiment_counts['Sentiment'] = pd.Categorical(sentiment_counts['Sentiment'], categories=filtered_order, ordered=True)
        sentiment_counts = sentiment_counts.sort_values('Sentiment')
        
        fig = px.bar(
            sentiment_counts,
            x="Sentiment",
            y="Count",
            title=f"Distribution of Sentiment Classifications - {product}",
            labels={"Sentiment": "Sentiment Category", "Count": "Number of Reviews"},
            color="Sentiment",
            color_discrete_map=filtered_colors
        )
        fig.update_layout(
            xaxis_title="Sentiment Category",
            yaxis_title="Number of Reviews",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
