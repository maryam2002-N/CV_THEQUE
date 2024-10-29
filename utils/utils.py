import re
import sqlite3

import streamlit as st
import fitz
import io
import os
import langdetect
from docx import Document
import re
from dateutil import parser
from datetime import datetime
from PIL import Image
import numpy as np
import pytesseract
from dateutil.relativedelta import relativedelta
from googletrans import Translator
from Model.models import *
import spacy
from spacy.matcher import Matcher
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tempfile
from streamlit_pdf_viewer import pdf_viewer
from docx2pdf import convert
import nltk
import spacy


nltk.download('punkt')
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
# Load models
rf_model = load_rf_model()
tfidf_vectorizer = load_vectorizer()
model_Doc2Vec = load_doc2vec_model()
# Prétraitement du texte
def preprocess_text(text):
    text = text.lower()
    text = re.sub('[^a-z]', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = ' '.join(text.split())
    return text

# Extraction du texte des fichiers
def extract_text_from_pdf(file):
    try:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du PDF : {str(e)}")

def extract_text_from_docx(file):
    try:
        doc = Document(io.BytesIO(file.read()))
        return "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du DOCX : {str(e)}")

# Function to extract text from an image using OCR
def extract_text_from_image(file):
    # Open the image
    image = Image.open(file)

    # Convert to RGB if it's not already
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Increase resolution
    image = image.resize((image.width * 2, image.height * 2), Image.LANCZOS)

    # Convert PIL Image to numpy array
    img_array = np.array(image)

    # Simple contrast enhancement
    img_array = np.clip(img_array * 1.2, 0, 255).astype(np.uint8)

    # Convert back to PIL Image
    enhanced_image = Image.fromarray(img_array)

    # Apply OCR with custom configuration
    custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    return pytesseract.image_to_string(enhanced_image, config=custom_config)

def standardize_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Fonction pour extraire le texte en fonction du type de fichier
def extract_text(file):
    try:
        file_extension = os.path.splitext(file.name)[1].lower()
        if file_extension == '.pdf':
            return standardize_text(extract_text_from_pdf(file))
        elif file_extension in ['.docx', '.doc']:
            return standardize_text(extract_text_from_docx(file))
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            return standardize_text(extract_text_from_image(file))
        else:
            raise ValueError(f"Type de fichier non supporté : {file_extension}")
    except Exception as e:
        raise ValueError(f"Erreur lors de l'extraction du texte : {str(e)}")
    return standardize_text(text)

# Fonction pour la traduction
def translate_to_french(text):
    if text is None or text.strip() == "":
        return ""
    try:
        translator = Translator()
        return translator.translate(text, src='en', dest='fr').text
    except Exception:
        return text  # Retourner le texte original si la traduction échoue


# Function to translate text to English
def translate_to_english(text):
    if not text.strip():
        st.warning("Texte vide fourni pour la traduction")
        return ""

    st.info(f"Traduction d'un texte de longueur {len(text)}. Échantillon : {text[:100]}...")

    try:
        # Attempt to encode and decode the text to catch any encoding issues
        text = text.encode('utf-8').decode('utf-8')
    except UnicodeEncodeError:
        st.warning("Le texte contient des caractères Unicode invalides. Tentative de nettoyage...")
        text = ''.join(char for char in text if ord(char) < 128)

    try:
        lang = langdetect.detect(text)
    except langdetect.LangDetectException:
        st.warning("Impossible de détecter la langue, supposons que ce n'est pas de l'anglais")
        lang = 'unknown'

    if lang == 'en':
        return text

    translator = Translator()
    chunk_size = 1000  # Reduced from 5000 to 1000
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    try:
        for chunk in chunks:
            try:
                translated_chunk = translator.translate(chunk, dest='en').text
                translated_chunks.append(translated_chunk)
            except Exception as e:
                st.warning(
                    f"Erreur lors de la traduction d'un morceau : {str(e)}. Utilisation du texte original pour ce morceau.")
                translated_chunks.append(chunk)

        return ' '.join(translated_chunks)
    except Exception as e:
        st.error(f"Erreur de traduction : {str(e)}")
        st.warning("Échec de la traduction. Utilisation du texte original.")
        return text


# Calcul de la similarité
def calculate_similarity(cv_text, jd_text):
    cv_vector = model_Doc2Vec.infer_vector(preprocess_text(cv_text).split())
    jd_vector = model_Doc2Vec.infer_vector(preprocess_text(jd_text).split())
    similarity = 100 * (np.dot(cv_vector, jd_vector)) / (norm(cv_vector) * norm(jd_vector))
    return round(similarity, 2)




import re
from datetime import datetime
from dateutil import parser
from dateutil.relativedelta import relativedelta



def extract_experience_durations(resume_text):
    date_patterns = [
        r'(\w+)\s*-\s*(\w+)\s+(\d{4})',  # For "mois - mois année"
        r'(\w+)-(\d{4})',  # For "mois-année"
        r'(\w+\s+\d{4})\s+to\s+(current|present|\w+\s+\d{4})',
        r'(\w{3,9}.?\s+\d{4})\s*-\s*(current|present|\w{3,9}.?\s+\d{4})',
        r'(\d{1,2}/\d{4})\s*-\s*(current|present|\d{1,2}/\d{4})',
        r'(\d{4})\s*-\s*(current|present|\d{4})',
        r'(\w{3,9}.?\s+\d{4})\s*–\s*(current|present|\w{3,9}.?\s+\d{4})',
        r'(\d{2}/\d{2}/\d{4})\s*-\s*(current|present|\d{2}/\d{2}/\d{4})',
        r'(\d{2}-\d{2}-\d{4})\s*-\s*(current|present|\d{2}-\d{2}-\d{4})',
        r'(\d{4}-\d{2}-\d{2})\s*-\s*(current|present|\d{4}-\d{2}-\d{2})',
        r'(\w+\s+\d{1,2},?\s+\d{4})\s*-\s*(current|present|\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}\s+\w+\s+\d{4})\s*-\s*(current|present|\d{1,2}\s+\w+\s+\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(current|present|\w+\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2})\s*-\s*(current|present|\d{1,2}/\d{1,2}/\d{2})',
        r'(Q[1-4]\s+\d{4})\s*-\s*(current|present|Q[1-4]\s+\d{4})',
        r'(\d{4})\s*-\s*(current|present|ongoing|\d{4})',
        r'(\d{2}/\d{2})\s*-\s*(current|present|\d{2}/\d{2})',
        r'(\d{2}\.\d{2}\.\d{4})\s*-\s*(current|present|\d{2}\.\d{2}\.\d{4})',
        r'(\d{4})\s*-\s*(current|present|\d{4})',  # For year-only formats
        r'(\d{4})\s*to\s*(current|present|\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(current|present|\w+\s+\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4})',
        r'(\w+)-\s*(\w+\s+\d{4})',  # e.g., "Avril- Juillet 2024"
        r'(\w+)\s+(\d{4})'
    ]
    experiences = []
    lines = resume_text.split('\n')
    current_section = None
    experience_start = -1

    for i, line in enumerate(lines):
        if re.search(r'WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EMPLOYMENT|CAREER HISTORY', line, re.IGNORECASE):
            current_section = "experience"
            experience_start = i
        elif re.search(r'EDUCATION|SKILLS|ADDITIONAL INFORMATION|ACHIEVEMENTS|PROJECTS', line, re.IGNORECASE):
            if current_section == "experience":
                break

        if current_section != "experience":
            continue

        for pattern in date_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                groups = match.groups()

                if len(groups) == 3:  # "mois - mois année" format
                    start_date = f"{groups[0]} {groups[2]}"
                    end_date = f"{groups[1]} {groups[2]}"
                elif len(groups) == 2 and not any(
                        keyword in groups[1].lower() for keyword in ['current', 'present', 'ongoing']):
                    if groups[0].isalpha() and groups[1].isdigit():  # "mois-année" format
                        start_date = f"{groups[0]} {groups[1]}"
                        end_date = f"{groups[0]} {groups[1]}"
                    else:
                        start_date, end_date = groups
                else:
                    start_date, end_date = groups

                if end_date.lower() in ['current', 'present', 'ongoing']:
                    end_date = datetime.now().strftime('%B %Y')

                try:
                    start = parser.parse(start_date, fuzzy=True, default=datetime(datetime.now().year, 1, 1))
                    end = parser.parse(end_date, fuzzy=True, default=datetime(datetime.now().year, 12, 31))

                    # Ensure end date is not before start date
                    if end < start:
                        end = end.replace(year=end.year + 1)

                    # Calculate the difference
                    diff = relativedelta(end, start)

                    # Calculate total months
                    total_months = diff.years * 12 + diff.months + (
                                diff.days / 30.44)  # 30.44 is the average number of days in a month

                    # Round to the nearest month
                    rounded_months = round(total_months)

                    duration = relativedelta(end, start)
                    years = duration.years
                    months = duration.months + (duration.days // 30)  # Approximate months

                    # Look for job title and company name
                    job_info = ""
                    company_info = ""
                    for j in range(i - 1, max(experience_start, i - 5), -1):
                        if lines[j].strip() and not any(re.search(p, lines[j]) for p in date_patterns):
                            if not job_info:
                                job_info = lines[j].strip()
                            elif not company_info:
                                company_info = lines[j].strip()
                                break

                    if not any(
                            exp['job_info'] == job_info and exp['company_info'] == company_info for exp in experiences):
                        years = rounded_months // 12
                        months = rounded_months % 12

                        experiences.append({
                            'date_range': match.group(),
                            'job_info': job_info,
                            'company_info': company_info,
                            'duration': f"{years} years, {months} months"
                        })

                except ValueError:
                    continue

    return experiences


def calculate_duration(start_date, end_date):
    try:
        start = parser.parse(start_date, fuzzy=True, default=datetime(datetime.now().year, 1, 1))
        end = parser.parse(end_date, fuzzy=True, default=datetime(datetime.now().year, 12, 31))

        if end < start:
            end = end.replace(year=end.year + 1)

        duration = relativedelta(end, start)
        years = duration.years
        months = duration.months + (duration.days // 30)

        return f"{years} years, {months} months"
    except:
        return "Durée non spécifiée"




def process_resume(resume_text):
    # Translate the entire resume to English
    translated_resume = translate_to_english(resume_text)

    # Extract experiences from the translated text
    experiences = extract_experience_durations(translated_resume)

    return experiences


def extract_personal_info(text):
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, text)

    # Phone number extraction (this pattern might need adjustment based on your specific needs)
    phone_pattern = r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"
    phones = re.findall(phone_pattern, text)


    # Name extraction (this is more complex and might require further refinement)
    name_pattern = r'\b([A-Z][a-z]+ [A-Z][a-z]+)\b'
    #names = re.findall(name_pattern, text)

    nlp1 = spacy.load('en_core_web_sm')
    matcher = Matcher(nlp1.vocab)

    # Define name patterns
    patterns = [
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name, Middle name, and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}],  # First name and Last name
        [{'POS': 'PROPN'}, {'POS': 'PROPN'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'PROPN'}],
        # First name, optional punctuation, Middle name, optional punctuation, and Last name
        [{'POS': 'PROPN'}, {'IS_PUNCT': True, 'OP': '?'}, {'POS': 'PROPN'}],
        # First name, optional punctuation, and Last name
    ]

    for pattern in patterns:
        matcher.add('NAME', patterns=[pattern])

    doc1 = nlp1(text)
    matches = matcher(doc1)

    for match_id, start, end in matches:
        span = doc1[start:end]
        # Adjusting to return the full name
        name = " ".join([token.text for token in span])

    return {
        'emails': emails,
        'phones': phones,
        'names': name
    }



# Function to generate word cloud
def generate_word_cloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    return plt


@st.cache_data
def load_skills_list():
    try:
        with open('utils/skills_list.txt', 'r') as f:
            return [line.strip().lower() for line in f]
    except FileNotFoundError:
        st.warning("Skills list file not found. Skill extraction will be skipped.")
        return []

def extract_skills(resume_text_fr, doc):
    skills_list = load_skills_list()
    skills_from_list = []
    if skills_list:
        for skill in skills_list:
            if re.search(r'\b' + re.escape(skill) + r'\b', resume_text_fr.lower()):
                skills_from_list.append(skill)

    skills_from_entities = [ent.text.lower() for ent in doc.ents if ent.label_ == "SKILLS"]

    return skills_from_list, skills_from_entities


def display_skills(skills):
    if not skills:
        st.info("No skills were extracted or skills list is unavailable.")
        return

    #st.markdown("##### Extracted Skills:")

    # Create a container for better control over styling
    skills_container = st.container()

    # Use custom CSS for styling
    skills_container.markdown("""
    <style>
    .skills-grid {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
        padding: 10px;
    }
    .skill-item {
        background-color: #e6f2ff;
        color: #333;
        padding: 5px 10px;
        border-radius: 15px;
        font-size: 14px;
        border: 1px solid #b3d9ff;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create the skills grid
    skills_html = "<div class='skills-grid'>"
    for skill in skills:
        skills_html += f"<div class='skill-item'>{skill.strip().upper()}</div>"
    skills_html += "</div>"

    skills_container.markdown(skills_html, unsafe_allow_html=True)


def extract_education(text_fr):
    education_keywords = [
        'éducation', 'formation', 'diplôme', 'baccalauréat', 'licence', 'master',
        'doctorat', 'bac', 'bts', 'dut', 'ingénieur', 'école', 'université'
    ]

    date_patterns = [

        r'(\w+)\s*-\s*(\w+)\s+(\d{4})',  # Pour "mois - mois année"
        r'(\w+)-(\d{4})',  # Pour "mois-année"
        r'(\w+\s+\d{4})\s+to\s+(current|present|\w+\s+\d{4})',
        r'(\w{3,9}.?\s+\d{4})\s*-\s*(current|present|\w{3,9}.?\s+\d{4})',
        r'(\d{1,2}/\d{4})\s*-\s*(current|present|\d{1,2}/\d{4})',
        r'(\d{4})\s*-\s*(current|present|\d{4})',
        r'(\w{3,9}.?\s+\d{4})\s*–\s*(current|present|\w{3,9}.?\s+\d{4})',
        r'(\d{2}/\d{2}/\d{4})\s*-\s*(current|present|\d{2}/\d{2}/\d{4})',
        r'(\d{2}-\d{2}-\d{4})\s*-\s*(current|present|\d{2}-\d{2}-\d{4})',
        r'(\d{4}-\d{2}-\d{2})\s*-\s*(current|present|\d{4}-\d{2}-\d{2})',
        r'(\w+\s+\d{1,2},?\s+\d{4})\s*-\s*(current|present|\w+\s+\d{1,2},?\s+\d{4})',
        r'(\d{1,2}\s+\w+\s+\d{4})\s*-\s*(current|present|\d{1,2}\s+\w+\s+\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(current|present|\w+\s+\d{4})',
        r'(\d{1,2}/\d{1,2}/\d{2})\s*-\s*(current|present|\d{1,2}/\d{1,2}/\d{2})',
        r'(Q[1-4]\s+\d{4})\s*-\s*(current|present|Q[1-4]\s+\d{4})',
        r'(\d{4})\s*-\s*(current|present|ongoing|\d{4})',
        r'(\d{2}/\d{2})\s*-\s*(current|present|\d{2}/\d{2})',
        r'(\d{2}.\d{2}.\d{4})\s*-\s*(current|present|\d{2}.\d{2}.\d{4})',
        r'(\d{4})\s*-\s*(current|present|\d{4})',  # Pour les formats année uniquement
        r'(\d{4})\sto\s(current|present|\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(current|present|\w+\s+\d{4})',
        r'(\w+\s+\d{4})\s*-\s*(\w+\s+\d{4})',
        r'(\w+)-\s*(\w+\s+\d{4})',  # e.g., "Avril- Juillet 2024"
        r'(\w+)\s+(\d{4})'
    ]

    education_entries = []
    lines = text_fr.split('\n')

    in_education_section = False
    current_entry = {}

    for i, line in enumerate(lines):
        line = line.strip().lower()

        # Vérifier si nous entrons dans la section éducation
        if any(keyword in line for keyword in education_keywords) and not in_education_section:
            in_education_section = True
            continue

        # Vérifier si nous sortons de la section éducation
        if in_education_section and any(keyword in line for keyword in ['expérience', 'compétences', 'projets']):
            in_education_section = False
            if current_entry:
                education_entries.append(current_entry)
                current_entry = {}
            break

        if in_education_section:
            # Rechercher les motifs de date
            for pattern in date_patterns:
                match = re.search(pattern, line, re.IGNORECASE)
                if match:
                    if current_entry:
                        education_entries.append(current_entry)
                        current_entry = {}

                    groups = match.groups()
                    if len(groups) == 3:  # format "mois - mois année"
                        current_entry['date_range'] = f"{groups[0]} {groups[2]} - {groups[1]} {groups[2]}"
                    elif len(groups) == 2:
                        if groups[1].lower() in ['current', 'present', 'ongoing']:
                            current_entry['date_range'] = f"{groups[0]} - Présent"
                        else:
                            current_entry['date_range'] = f"{groups[0]} - {groups[1]}"
                    else:
                        current_entry['date_range'] = groups[0]

                    # Rechercher les informations sur le diplôme dans la ligne suivante
                    if i + 1 < len(lines):
                        current_entry['degree'] = lines[i + 1].strip()
                    break

            # Si nous avons une entrée courante mais pas encore de diplôme, ajouter cette ligne comme information supplémentaire
            if current_entry and 'degree' in current_entry:
                if 'additional_info' not in current_entry:
                    current_entry['additional_info'] = line
                else:
                    current_entry['additional_info'] += ' ' + line

    # Ajouter la dernière entrée si elle existe
    if current_entry:
        education_entries.append(current_entry)

    return education_entries


import logging

logging.basicConfig(level=logging.DEBUG)


def display_pdf(file):
    try:
        logging.debug(f"Attempting to read file: {file.name}")
        binary_data = file.getvalue()  # Utiliser getvalue() au lieu de read()
        logging.debug(f"File read successfully. Size: {len(binary_data)} bytes")

        # Utiliser le paramètre 'input' explicitement
        pdf_viewer(input=binary_data, width=700)
        logging.debug("PDF displayed using streamlit-pdf-viewer")
    except Exception as e:
        logging.error(f"Error in display_pdf: {str(e)}", exc_info=True)
        st.error(f"Erreur lors de l'affichage du PDF : {str(e)}")
        st.warning("L'affichage du PDF a échoué. Le texte extrait sera affiché à la place.")
        text = extract_text_from_pdf(file)
        st.text_area("Texte extrait du CV", text, height=300)

def display_image(file):
    st.image(file, use_column_width=True)


def convert_docx_to_pdf(docx_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_docx = os.path.join(temp_dir, "temp.docx")
        temp_pdf = os.path.join(temp_dir, "temp.pdf")

        with open(temp_docx, "wb") as f:
            f.write(docx_file.getvalue())

        convert(temp_docx, temp_pdf)

        with open(temp_pdf, "rb") as f:
            pdf_bytes = f.read()

    return pdf_bytes

def categorize_cv(text):
    # Transform the text using the loaded vectorizer
    text_vectorized = tfidf_vectorizer.transform([text])

    # Predict the category
    category = rf_model.predict(text_vectorized)[0]

    return category




def get_all_users():
    print("Attempting to get all users...")
    try:
        with sqlite3.connect('db_cv_thèque.db') as conn:
            print("Connected to database successfully")
            c = conn.cursor()
            c.execute("SELECT id, username, password, role, nom, prenom FROM users")
            users = c.fetchall()
            print(f"Retrieved {len(users)} users")
        return users
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        return []

def delete_user(user_id):
    with sqlite3.connect('cv_database2.db') as conn:
        c = conn.cursor()
        c.execute("DELETE FROM users WHERE id = ?", (user_id,))
        conn.commit()