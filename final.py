from urllib.request import urlopen
from bs4 import BeautifulSoup
from pymongo import MongoClient
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import re
import string

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['faculty_data']
faculty_collection = db['faculty_pages']
index_terms_collection = db['index_terms']

# Helper function to fetch URL content
def retrieve_url(url):
    try:
        response = urlopen(url)
        return response.read()
    except Exception as e:
        return None  # Silently return None if the request fails

# Preprocess text: remove stop words, lemmatize, and clean
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = text.split()
    processed_tokens = []

    # Remove stopwords and lemmatize tokens
    for token in tokens:
        token = token.lower().strip(string.punctuation)
        
        # Exclude stop words and non-alphanumeric tokens
        if token and token not in stop_words and not token.isdigit():
            lemmatized_token = lemmatizer.lemmatize(token)
            processed_tokens.append(lemmatized_token)

    return " ".join(processed_tokens)

# Parse the "About Me" and accolades sections from individual profile pages
def parse_profile_page(profile_url):
    html = retrieve_url(profile_url)
    if not html:
        return None

    soup = BeautifulSoup(html, 'html.parser')

    # Extract the "About Me" section
    about_section = soup.find("div", class_="fac-staff")
    about_text = None
    if about_section:
        paragraphs = about_section.find_all("p")
        about_text = " ".join([p.get_text(strip=True) for p in paragraphs])

    # Extract faculty accolades
    accolades = {}
    accolades_section = soup.find("aside", class_="span3 fac rightcol")
    if accolades_section:
        accolades_divs = accolades_section.find_all("div", class_="accolades")
        for div in accolades_divs:
            header = div.find("h2").text.strip() if div.find("h2") else None
            content = []
            for element in div.find_all(["p", "a"]):
                if element.name == "a":
                    content.append(f"{element.text.strip()} ({element.get('href')})")
                elif element.name == "p":
                    content.append(element.text.strip())
            if header:
                accolades[header] = " ".join(content)

    return {
        "about": about_text,
        "accolades": accolades
    }

# Parse the HTML to extract faculty information
def parse_faculty_card(html):
    soup = BeautifulSoup(html, 'html.parser')
    cards = soup.find_all("div", class_="card h-100")
    faculty_links = []

    for card in cards:
        # Extract professor name
        name_tag = card.find("h3")
        name = name_tag.text.strip() if name_tag else "N/A"
        
        # Extract profile link
        profile_tag = card.find("a", {"aria-label": lambda x: x and "open" in x})
        profile_link = profile_tag.get('href') if profile_tag else None
        if profile_link:
            profile_link = f"https://www.cpp.edu{profile_link}"
        
        # Extract phone number
        phone_tag = card.find("span", class_="sr-only", string="phone number or extension")
        phone = phone_tag.find_next_sibling(string=True).strip() if phone_tag else "N/A"
        
        # Extract office location
        office_tag = card.find("span", class_="sr-only", string="office location")
        office = office_tag.find_next_sibling(string=True).strip() if office_tag else "N/A"
        
        # Extract email
        email_tag = card.find("a", {"aria-label": lambda x: x and "email" in x})
        email = email_tag['href'].replace("mailto:", "") if email_tag else "N/A"
        
        # Store extracted data
        faculty_links.append({
            "name": name,
            "profile_link": profile_link,
            "phone": phone,
            "office": office,
            "email": email
        })

    return faculty_links

# Crawl from seed URL to target URL and execute the extraction
def crawl_department(seed_url):
    html = retrieve_url(seed_url)
    if not html:
        return
    
    # Parse the seed URL to find links
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a", href=True)
    target_url = None
    for link in links:
        href = link['href']
        if re.search(r'faculty-staff/index\.shtml', href):
            target_url = f"https://www.cpp.edu{href}" if href.startswith('/') else href
            break

    if not target_url:
        return

    # Fetch and parse the target URL
    target_html = retrieve_url(target_url)
    if not target_html:
        return

    faculty_data = parse_faculty_card(target_html)
    
    # Visit each profile link and extract additional information
    for faculty in faculty_data:
        profile_url = faculty["profile_link"]
        profile_data = parse_profile_page(profile_url)
        if profile_data:
            faculty.update(profile_data)  # Add additional information to the faculty data
        faculty_collection.insert_one(faculty)

# Generate TF-IDF matrix for documents
def calculate_tfidf_matrix(documents):
    vectorizer = TfidfVectorizer(ngram_range=(1, 3))  # Using unigrams, bigrams, and trigrams
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

# Populate the index terms collection with preprocessed terms (stop words removed, lemmatized)
def create_index_terms_collection(vectorizer, tfidf_matrix, professor_data):
    feature_names = vectorizer.get_feature_names_out()
    for i, professor in enumerate(professor_data):
        tfidf_vector = tfidf_matrix[i].toarray().flatten()
        non_zero_indices = tfidf_vector.nonzero()[0]
        
        # Preprocess terms before storing them
        terms_with_tfidf = [
            {
                "term": preprocess_text(feature_names[idx]),  # Preprocess each term
                "tfidf": tfidf_vector[idx],
                "professor_name": professor["name"],
                "professor_id": professor["id"],
                "profile_link": professor["profile_link"]
            }
            for idx in non_zero_indices
        ]
        if terms_with_tfidf:
            index_terms_collection.insert_many(terms_with_tfidf)

# Fetch professor data and prepare TF-IDF
def prepare_professor_data():
    documents = []
    professor_data = []

    for professor in faculty_collection.find():
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
    processed_query = preprocess_text(query)
    query_vector = vectorizer.transform([processed_query])
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]

    seen = set()
    results = []

    for idx in ranked_indices:
        professor = professor_data[idx]
        name = professor["name"]
        profile_link = professor["profile_link"]

        if (name, profile_link) not in seen:
            seen.add((name, profile_link))
            first_paragraph = professor["about"].split("\n")[0] if professor["about"] else "No about section available"
            truncated_about = " ".join(first_paragraph.split()[:50]) + "..." if len(first_paragraph.split()) > 50 else first_paragraph

            results.append({
                "name": name,
                "profile_link": profile_link,
                "about_first_paragraph": truncated_about,
                "similarity_score": similarities[idx]
            })

        if len(results) == 5:
            break

    if all(result["similarity_score"] == 0 for result in results):
        return [{"name": "No relevant document", "profile_link": "", "about_first_paragraph": "", "similarity_score": 0}]

    return results

# Main Execution
if __name__ == "__main__":
    seed_url = "https://www.cpp.edu/cba/international-business-marketing/index.shtml"
    crawl_department(seed_url)

    documents, professor_data = prepare_professor_data()
    vectorizer, tfidf_matrix = calculate_tfidf_matrix(documents)
    create_index_terms_collection(vectorizer, tfidf_matrix, professor_data)

    user_query = input("Enter your search query: ")
    results = search_professors(user_query)

    for result in results:
        print(f"Professor: {result['name']}")
        print(f"Profile Link: {result['profile_link']}")
        print(f"About (Truncated to 50 Words): {result['about_first_paragraph']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print("-" * 80)
