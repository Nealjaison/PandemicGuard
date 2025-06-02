import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required nltk data (run once)
nltk.download('stopwords')
nltk.download('wordnet')

def clean_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_text_column(df, column):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def process(text):
        if pd.isnull(text):
            return ""
        cleaned = clean_text(text)
        tokens = cleaned.split()
        filtered = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        return " ".join(filtered)

    df[column] = df[column].apply(process)
    return df

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="NLP Preprocessing for PandemicGuard")
    parser.add_argument('--input', type=str, required=True, help="Input CSV file")
    parser.add_argument('--output', type=str, required=True, help="Output CSV file with preprocessed text")
    parser.add_argument('--text_column', type=str, required=True, help="Name of the text column to preprocess")

    args = parser.parse_args()

    data = pd.read_csv(args.input)
    data = preprocess_text_column(data, args.text_column)
    data.to_csv(args.output, index=False)
    print(f"Preprocessed NLP data saved to {args.output}")
