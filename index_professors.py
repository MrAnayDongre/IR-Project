from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['faculty_data']
collection = db['faculty_pages']

# Preprocess text: remove stop words, lemmatize, and clean
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    processed_tokens = []
    for token in tokens:
        token = token.lower().strip(string.punctuation)
        if token and token not in stop_words:
            lemmatized_token = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized_token)
    return " ".join(processed_tokens)

# Generate TF-IDF matrix for documents
def calculate_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Using unigrams, bigrams, and trigrams
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Fetch professor data and prepare TF-IDF
def prepare_professor_data():
    documents = []
    professor_data = []

    for professor in collection.find():
        text = professor.get('about', '') + " " + " ".join(professor.get('accolades', {}).values())
        if text.strip():
            processed_text = preprocess_text(text)
            documents.append(processed_text)
            professor_data.append({
                "id": professor["_id"],
                "name": professor["name"],
                "profile_link": professor["profile_link"],
                "about": professor.get("about", "")
            })

    return documents, professor_data

# Handle user query and return ranked professors
def search_professors(query):
    # Preprocess query
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])

    # Calculate similarity
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]

    # Return top results
    results = []
    for idx in ranked_indices[:5]:  # Top 5 results
        professor = professor_data[idx]
        first_paragraph = professor["about"].split("\n")[0] if professor["about"] else "No about section available"
        results.append({
            "name": professor["name"],
            "profile_link": professor["profile_link"],
            "about_first_paragraph": first_paragraph,
            "similarity_score": similarities[idx]
        })
    return results

# Main Execution
if __name__ == "__main__":
    # Prepare data
    documents, professor_data = prepare_professor_data()

    # Generate TF-IDF matrix
    vectorizer, tfidf_matrix = calculate_tfidf_matrix(documents)

    # Example query
    user_query = input("Enter your search query: ")
    results = search_professors(user_query)

    # Display results
    for result in results:
        print(f"Professor: {result['name']}")
        print(f"Profile Link: {result['profile_link']}")
        print(f"About (First Paragraph): {result['about_first_paragraph']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("-" * 80)
