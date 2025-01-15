import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from collections import Counter
import emoji
from preprocess import preprocess_dataset1, preprocess_dataset2

# Load Data
def load_data():
    """
    Load the WhatsApp chat data from the CSV files.

    Returns:
        df1 (pd.DataFrame): The preprocessed Dataset 1.
        df2 (pd.DataFrame): The preprocessed Dataset 2.
    """
    try:
        # Load Dataset 1
        df1 = pd.read_csv("D:/Portfolio Projects/WhatsApp Chat Analysis/data/whatsapp_chat_analysis/Cleaned_data.csv")
        df1 = preprocess_dataset1(df1)
        
        # Load Dataset 2
        df2 = pd.read_csv("D:/Portfolio Projects/WhatsApp Chat Analysis/data/whatsapp_chat/Cleaned_data.csv")
        df2 = preprocess_dataset2(df2)
        
        return df1, df2
    
    except FileNotFoundError as e:
        st.error(f"Error: {e}. Please ensure that the file exists.")
        return None, None
    
    except pd.errors.EmptyDataError as e:
        st.error(f"Error: {e}. The file is empty.")
        return None, None
    
    except pd.errors.ParserError as e:
        st.error(f"Error: {e}. There was an error parsing the file.")
        return None, None
    
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None, None

df1, df2 = load_data()

if df1 is not None and df2 is not None:
    # Proceed with the analysis
    pass
else:
    st.error("Error loading datasets. Please check the file paths and try again.")

df1, df2 = load_data()


#Sidebar for navigatio
st.sidebar.title("Navigation")
analysis_type = st.sidebar.radio(
    "Choose Analysis",
    (
        "Dataset 1 Overview",
        "Dataset 1 Top Users",
        "Dataset 1 Activity Trends",
        "Dataset 1 Word Cloud",
        "Dataset 1 Emoji Analysis",
        "Dataset 1 Sentiment Analysis",
        "Dataset 2 Overview",
        "Dataset 2 Top Users",
        "Dataset 2 Activity Trends"
    )
)

# 1. Overview Section
if analysis_type == "Dataset 1 Overview":
    """
    Display an overview of the WhatsApp chat data, including the total number of messages, unique users, and time range.
    """
    st.title("WhatApp Chat Analysis Overview")
    st.write("### General Statistics")
    st.write(f"Total Messages: {df1.shape[0]}")
    st.write(f"Total Users: {df1['user'].nunique()}")
    st.write(f"Time Range: {df1['year'].min()} - {df1['year'].max()}")

# 2. Top User Analysis
elif analysis_type == "Dataset 1 Top Users":
    """
    Display the top 10 most active users in the WhatsApp chat data.
    """
    st.title("Top 10 Most Active Users")
    top_10_users = df1['user'].value_counts().head(10)
    st.bar_chart(top_10_users)

# 3. Activity Trends Section
elif analysis_type =="Dataset 1 Activity Trends":
    """
    Display the activity trends in the WhatsApp chat data, including hourly, daily, and monthly activity.
    """
    st.title("Activity Trends")

    # Hourly Activity
    hourly_activity = df1.groupby("hour").size()
    st.write("### Hourly Activity")
    fig,ax = plt.subplots(figsize=(10,5))
    sns. barplot(x=hourly_activity.index,y=hourly_activity.values,palette='viridis', ax=ax)
    ax.set_title("Messages Sent by Hour")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Number of Messages")
    st.pyplot(fig)

# 4. WordCloud Section
elif analysis_type == "Dataset 1 Word Cloud":
    """
    Display a word cloud of the most frequently used words in the WhatsApp chat data.
    """
    st.title("Word Cloud")
    all_messages = " ".join(df1['message'].dropna())
    wordcloud = WordCloud(width=800,height=400,background_color='white').generate(all_messages)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)


# 5. Emoji Analysis Section
elif analysis_type == "Dataset 1 Emoji Analysis":
    """
    Display the top 10 most frequently used emojis in the WhatsApp chat data.
    """
    st.title("Emoji Analysis")

    # Extract Emojis
    def extract_emojis(text):
        return " ".join([c for c in text if c in emoji.EMOJI_DATA])
    
    df1['emojis'] = df1['message'].apply(lambda x: extract_emojis(x) if isinstance(x, str) else '')
    all_emojis = ''.join(df1['emojis'])
    emoji_count = Counter(all_emojis)
    emoji_df = pd.DataFrame(emoji_count.items(), columns=['emoji', 'count']).sort_values(by='count', ascending=False).head(10)
    
    # Display emojis
    st.write("### Top 10 Most Used Emojis")
    st.dataframe(emoji_df)
    
    # Plot emojis
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='emoji', y='count', data=emoji_df, palette='coolwarm', ax=ax)
    ax.set_title("Top Emojis")
    st.pyplot(fig)

# 6. Sentiment Analysis Section
elif analysis_type == "Dataset 1 Sentiment Analysis":
    """
    Perform sentiment analysis on the WhatsApp chat data and display the results.
    """
    st.title("Sentiment Analysis")
    from textblob import TextBlob
    
    # Sentiment Polarity
    df1['polarity'] = df1['message'].apply(lambda x: TextBlob(x).sentiment.polarity if isinstance(x, str) else 0)
    df1['sentiment'] = df1['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
    
    # Sentiment Counts
    sentiment_counts = df1['sentiment'].value_counts()
    st.write("### Sentiment Distribution")
    st.bar_chart(sentiment_counts)

elif analysis_type == "Dataset 2 Overview":
    """
    Display an overview of the second dataset, including total messages and unique users.
    """
    st.title("Dataset 2 Overview")
    st.write("### General Statistics")
    st.write(f"Total rows: {df2.shape[0]}")
    st.write(f"Unique Users: {df2['names'].nunique()}")
    duplicates_df2 = df2.duplicated().sum()
    duplicate_percentage_df2 = (duplicates_df2/df2.shape[0])*100
    st.write(f"Duplicated rows: {duplicates_df2} ({duplicate_percentage_df2:.2f}%)")

elif analysis_type == "Dataset 2 Top Users":
    """
    Display the top 10 most active users in the second dataset.
    """
    st.title("Top 10 Most Active Users (Dataset 2)")
    top_10_users_df2=df2['names'].value_counts().head(10).reset_index()
    top_10_users_df2.columns=['user',"message count"]
    st.bar_chart(top_10_users_df2.set_index("user"))

elif analysis_type == "Dataset 2 Activity Trends":
    """
    Display the activity trends in the second dataset, including hourly activity.
    """
    st.title("Activity Trends (Dataset 2)")    

    # Hourly Activity
    hourly_activity_df2=df2.groupby("hours").size()
    st.write("### Hourly Activity")
    fig, ax=plt.subplots(figsize=(10,5))
    sns.barplot(x=hourly_activity_df2.index,y=hourly_activity_df2.values,palette='viridis', ax=ax)
    ax.set_title("Messages Sent by Hour")
    ax.set_xlabel("Hours of Day")
    ax.set_ylabel("Number of Messages")
    st.pyplot(fig)

    # Daily Activity
    st.write("### Daily Activity")
    df2['day_of week']=df2['datetime'].dt.day_name()
    days_order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_activity_df2=df2['day_of_week'].value_counts().reindex(days_order)
    fig, ax=plt.subplots(figsize=(10,6))
    sns.barplot(x=daily_activity_df2.index,y=daily_activity_df2.values,palette='magma',ax=ax)
    ax.set_title("Message Sent by Day of the Week")
    ax.set_xlabel("Day of the Week")
    ax.set_ylabel("Number of Messages")
    st.pyplot(fig)

    # Monthly Activity
    st.write("### Monthly Activity")
    df2['month'] = df2['datetime'].dt.month_name()
    months_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    monthly_activity_df2 = df2['month'].value_counts().reindex(months_order)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x=monthly_activity_df2.index, y=monthly_activity_df2.values, palette='magma', ax=ax)
    ax.set_title("Messages Sent by Month")
    ax.set_xlabel("Month")
    ax.set_ylabel("Number of Messages")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    st.pyplot(fig)
# Heatmap of Day vs Hour
    st.write("### Heatmap of Messages by Day and Hour")
    heatmap_data = df2.pivot_table(index='day_of_week', columns='hour', values='datetime', aggfunc='count', fill_value=0).reindex(days_order)
    fig, ax = plt.subplots(figsize=(15, 7))
    sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='d', ax=ax)
    ax.set_title("Heatmap of Messages by Day and Hour")
    ax.set_xlabel("Hour of the Day")
    ax.set_ylabel("Day of the Week")
    st.pyplot(fig)

# Footer
st.sidebar.markdown("Developed by Durdana Khalid")