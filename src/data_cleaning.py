
import pandas as pd
import re
import spacy

# Loading English model spaCy
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    """
    Cleans the text data by removing special characters, applying spaCy for tokenization, 
    lemmatization, and removing stopwords. Keeps only alphabetic words.
    """
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', str(text))  # Ensure text is a string
    text = text.lower()

    # Process text with spaCy
    doc = nlp(text)

    # Lemmatize, remove stopwords, and keep only alphabetic words
    tokens = [
        token.lemma_ for token in doc
        if not token.is_stop and token.is_alpha
    ]

    return ' '.join(tokens)

def clean_product_name(product_name):
    """
    Cleans the product name by removing special characters and converting to lowercase.
    """
    product_name = re.sub(r'[^a-zA-Z\s]', '', str(product_name))  # Ensure product_name is a string
    return product_name.lower()

def clean_data(input_path, output_path):
    """
    Cleans the dataset by processing review_text, product_name, and other relevant fields.
    """
    print("Loading data...")
    df = pd.read_csv(input_path)

    # Remove missing values
    print("Removing missing values...")
    df = df.dropna(subset=['review_text', 'is_recommended', 'brand_name', 'price_usd', 'submission_time', 'product_name'])

    # Convert submission_time to datetime
    print("Processing dates...")
    df['submission_time'] = pd.to_datetime(df['submission_time'], errors='coerce')
    df['year'] = df['submission_time'].dt.year
    df['month'] = df['submission_time'].dt.month

    # Calculate review length
    print("Calculating text length...")
    df['review_length'] = df['review_text'].apply(len)

    # Clean review text
    print("Cleaning review text...")
    df['cleaned_review_text'] = df['review_text'].apply(clean_text)

    # Clean product names
    print("Cleaning product names...")
    df['cleaned_product_name'] = df['product_name'].apply(clean_product_name)

    # Save cleaned data
    print("Saving cleaned data...")
    df.to_csv(output_path, index=False)
    print(f"Data cleaning completed. Cleaned data saved to {output_path}")

if __name__ == "__main__":
    # File paths
    raw_data_path = "data/reviews_0-250.csv"
    cleaned_data_path = "data/cleaned_reviews.csv"

    # Call the cleaning function
    clean_data(raw_data_path, cleaned_data_path)






