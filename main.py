from urllib.request import urlopen
from bs4 import BeautifulSoup
from pymongo import MongoClient
import re

# MongoDB setup
client = MongoClient('mongodb://localhost:27017/')
db = client['faculty_data']
collection = db['faculty_pages']

# Helper function to fetch URL content
def retrieve_url(url):
    try:
        response = urlopen(url)
        return response.read()
    except Exception as e:
        print(f"Failed to retrieve {url}: {e}")
        return None

# Parse the "About Me" and accolades sections from individual profile pages
def parse_profile_page(profile_url):
    html = retrieve_url(profile_url)
    if not html:
        print(f"Failed to retrieve profile URL: {profile_url}")
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
        phone_tag = card.find("span", class_="sr-only", text="phone number or extension")
        phone = phone_tag.find_next_sibling(text=True).strip() if phone_tag else "N/A"
        
        # Extract office location
        office_tag = card.find("span", class_="sr-only", text="office location")
        office = office_tag.find_next_sibling(text=True).strip() if office_tag else "N/A"
        
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
def crawl_department(seed_url, target_url=None):
    print(f"Starting crawl from seed URL: {seed_url}")
    html = retrieve_url(seed_url)
    if not html:
        print("Failed to retrieve seed URL")
        return
    
    # Parse the seed URL to find links
    soup = BeautifulSoup(html, 'html.parser')
    links = soup.find_all("a", href=True)
    for link in links:
        href = link['href']
        # Match specifically for the correct target URL
        if re.search(r'faculty-staff/index\.shtml', href):
            target_url = f"https://www.cpp.edu{href}" if href.startswith('/') else href
            print(f"Target URL found: {target_url}")
            break

    if not target_url:
        print("Could not find the correct target URL")
        return

    # Fetch and parse the target URL
    target_html = retrieve_url(target_url)
    if not target_html:
        print("Failed to retrieve target URL")
        return

    faculty_data = parse_faculty_card(target_html)
    
    # Visit each profile link and extract additional information
    for faculty in faculty_data:
        profile_url = faculty["profile_link"]
        profile_data = parse_profile_page(profile_url)
        if profile_data:
            faculty.update(profile_data)  # Add additional information to the faculty data
        collection.insert_one(faculty)
        print(f"Stored: {faculty}")

# Seed URL
seed_url = "https://www.cpp.edu/cba/international-business-marketing/index.shtml"

# Execute the crawl
crawl_department(seed_url)
