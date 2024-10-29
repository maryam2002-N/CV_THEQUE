import sys
import os
import plotly.express as px
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import concurrent.futures
from utils.utils import *
from Model.models import *
import bcrypt
from streamlit_lottie import st_lottie
import requests
import base64
import tempfile
import os
from PIL import Image
import io


# Configuration de l'authentification

# Définir un répertoire pour stocker les fichiers
UPLOAD_DIRECTORY = "uploaded_cvs"

# Assurez-vous que le répertoire existe
if not os.path.exists(UPLOAD_DIRECTORY):
    os.makedirs(UPLOAD_DIRECTORY)

# Initialisation de la base de données
conn = sqlite3.connect('db_cv_thèque.db')
c = conn.cursor()

# Création des tables si elles n'existent pas
c.execute('''CREATE TABLE IF NOT EXISTS cvs
             (id INTEGER PRIMARY KEY, filename TEXT, category TEXT, note INTEGER, extracted_info TEXT, file_path TEXT, user_id INTEGER)''')

c.execute('''CREATE TABLE IF NOT EXISTS users
             (id INTEGER PRIMARY KEY,nom TEXT, prenom TEXT, username TEXT UNIQUE, password TEXT, role TEXT)''')

c.execute('''CREATE TABLE IF NOT EXISTS poste_similarite
             (id INTEGER PRIMARY KEY, cv_id INTEGER, poste_id INTEGER, score REAL, date TIMESTAMP,
             FOREIGN KEY (cv_id) REFERENCES cvs(id),
             FOREIGN KEY (poste_id) REFERENCES poste(id))''')

c.execute('''CREATE TABLE IF NOT EXISTS poste
             (id INTEGER PRIMARY KEY, poste_intitulé TEXT, description_poste TEXT)''')

conn.commit()

# Fonctions d'authentification et de gestion de session


def init_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'user_role' not in st.session_state:
        st.session_state.user_role = None
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if "choice" not in st.session_state:
        st.session_state.choice = "Accueil"

def authenticate(username, password):
    c.execute("SELECT id, password, role FROM users WHERE username = ?", (username,))
    result = c.fetchone()

    if result:
        user_id, stored_password, role = result
        if bcrypt.checkpw(password.encode('utf-8'), stored_password.encode('utf-8')):
            return True, role, user_id
    return False, None, None


def sidebar_content():
    st.sidebar.image(r"C:\Users\HP\PycharmProjects\cv_thèque_ctm\images\ctm_logo.png", width=100)
    st.sidebar.title("Analyseur de CV")

    if not st.session_state.authenticated:
        username = st.sidebar.text_input("Nom d'utilisateur")
        password = st.sidebar.text_input("Mot de passe", type="password")
        login_button = st.sidebar.button("Se connecter")

        if login_button:
            authenticated, role, user_id = authenticate(username, password)
            if authenticated:
                st.session_state.authenticated = True
                st.session_state.username = username
                st.session_state.user_role = role
                st.session_state.user_id = user_id
                st.sidebar.success("Connexion réussie!")
                st.rerun()
            else:
                st.sidebar.error("Nom d'utilisateur ou mot de passe incorrect")
    else:
        st.sidebar.title(f"Bienvenue, {st.session_state.username}!")
        menu = ["Analyser CVs", "Tableau de Bord"]
        if st.session_state.user_role == 'admin':
            menu.append("Gestion des Utilisateurs")
        choice = st.sidebar.selectbox("Menu", menu)

        if st.sidebar.button("Se déconnecter"):
            logout()

        return choice


def styled_sidebar():

    choice = None
    with st.sidebar:
        st.image(r"C:\Users\HP\PycharmProjects\cv_thèque_ctm\images\ctm_logo.png", width=100)

        st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: white;
        }
        .sidebar .sidebar-content .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        .sidebar .sidebar-content .stButton > button {
            width: 100%;
            background-color: #ff4b4b;
            color: white;
            border: none;
            padding: 0.5rem 1rem;
            margin: 0.25rem 0;
            border-radius: 4px;
        }
        .sidebar .sidebar-content .stSelectbox [data-baseweb="select"] {
            background-color: white;
            color: black;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        </style>
        """, unsafe_allow_html=True)

        st.title("CV THÈQUE")

        if not st.session_state.authenticated:
            username = st.text_input("Nom d'utilisateur")
            password = st.text_input("Mot de passe", type="password")
            login_button = st.button("Se connecter")

            if login_button:
                authenticated, role, user_id = authenticate(username, password)
                if authenticated:
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.session_state.user_role = role
                    st.session_state.user_id = user_id
                    st.success("Connexion réussie!")
                    st.rerun()
                else:
                    st.error("Nom d'utilisateur ou mot de passe incorrect")
        else:
            st.write(f"Bienvenue, {st.session_state.username}!")

            menu =["Accueil", "Analyser CVs", "Tableau de Bord", "Filtrage CV", "Filtrage Postes"]
            if st.session_state.user_role == 'admin':
                menu.append("Gestion des Utilisateurs")

            choice = st.selectbox("", menu, key="sidebar_menu")
            st.session_state.choice = choice  # Mise à jour de st.session_state.choice

            if st.button("Se déconnecter"):
                logout()

        return st.session_state.choice

def display_filtered_cvs(cvs):
    columns = ['ID', 'Nom du fichier', 'Catégorie', 'Note','description_poste', 'Score de similarité']
    data = [[cv[0], cv[1], cv[2], cv[3], cv[8] if cv[8] != 'NULL' else 'N/A',cv[7]] for cv in cvs]
    df = pd.DataFrame(data, columns=columns)
    st.dataframe(df)

    for cv in cvs:
        with st.expander(f"Détails pour {cv[1]}"):
            st.write(f"Catégorie : {cv[2]}")
            st.write(f"Note : {cv[3]}")
            st.write(f"description_poste : {cv[8] if cv[8] != 'NULL' else 'N/A'}")
            st.write(f"Score de similarité : {cv[7]}")
            try:
                extracted_info = eval(cv[4])
                st.json(extracted_info)
            except:
                st.text(cv[4])

def filtrage_cv():
    st.markdown("""
        <style>
        .input-section .stButton {
            margin-top: 1rem;
            display: flex;
            justify-content: center;
        }
        .centered-button {
            display: flex;
            justify-content: center;
        }
        </style>
        """, unsafe_allow_html=True)
    st.header("Filtrage des CV")

    # Récupérer toutes les catégories uniques
    all_categories = ["Tous"] + list(set([cv[2] for cv in get_cvs()]))

    # Filtres
    category_filter = st.selectbox("Filtrer par catégorie", all_categories)
    min_score = st.slider("Score minimum de similarité", 0.0, 100.0, 0.0, 0.1)
    min_note = st.slider("Note minimum", 0, 100, 0, 1)
    # Bouton pour appliquer les filtres
    if st.button("Appliquer les filtres"):
        filtered_cvs = get_filtered_cvs(category_filter, min_score, min_note)

        if filtered_cvs:
            st.success(f"{len(filtered_cvs)} CV(s) trouvé(s)")
            display_filtered_cvs(filtered_cvs)
        else:
            st.info("Aucun CV ne correspond aux critères de filtrage.")

def get_filtered_cvs(category, min_score, min_note):
    query = """
    SELECT cvs.*, poste_similarite.score, poste.poste_intitulé, poste.description_poste
    FROM cvs
    LEFT JOIN poste_similarite ON cvs.id = poste_similarite.cv_id
    LEFT JOIN poste ON poste_similarite.poste_id = poste.id
    WHERE cvs.note >= ?
    AND (poste_similarite.score >= ? OR poste_similarite.score IS NULL)
    """
    params = [min_note, min_score]

    if category != "Tous":
        query += " AND cvs.category = ?"
        params.append(category)

    if st.session_state.user_role != 'admin':
        query += " AND cvs.user_id = ?"
        params.append(st.session_state.user_id)

    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute(query, params)
        return c.fetchall()

def filtrage_postes():
    st.header("Filtrage des Postes")

    # Récupérer tous les intitulés de postes uniques
    all_postes = get_all_postes()

    # Filtre
    poste_filter = st.selectbox("Filtrer par intitulé de poste", ["Tous"] + all_postes)

    # Bouton pour appliquer le filtre
    if st.button("Appliquer le filtre"):
        if poste_filter == "Tous":
            filtered_postes = get_all_postes_details()
        else:
            filtered_postes = get_filtered_postes(poste_filter)

        if filtered_postes:
            for poste in filtered_postes:
                with st.expander(f"Poste: {poste['intitule']}"):
                    st.write(f"Description: {poste['description']}")
                    st.write("CVs correspondants:")
                    for cv in poste['cvs']:
                        st.write(f"- {cv['filename']} (Score: {cv['score']})")
        else:
            st.info("Aucun poste ne correspond aux critères de filtrage.")

def get_all_postes():
    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute("SELECT DISTINCT poste_intitulé FROM poste")
        return [row[0] for row in c.fetchall()]

def get_all_postes_details():
    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute("""
            SELECT p.poste_intitulé, p.description_poste, c.filename, ps.score
            FROM poste p
            LEFT JOIN poste_similarite ps ON p.id = ps.poste_id
            LEFT JOIN cvs c ON ps.cv_id = c.id
            ORDER BY p.poste_intitulé, ps.score DESC
        """)
        rows = c.fetchall()

    postes = {}
    for row in rows:
        intitule, description, filename, score = row
        if intitule not in postes:
            postes[intitule] = {'intitule': intitule, 'description': description, 'cvs': []}
        if filename and score is not None:
            postes[intitule]['cvs'].append({'filename': filename, 'score': score})

    return list(postes.values())

def get_filtered_postes(poste_intitule):
    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute("""
            SELECT p.poste_intitulé, p.description_poste, c.filename, ps.score
            FROM poste p
            LEFT JOIN poste_similarite ps ON p.id = ps.poste_id
            LEFT JOIN cvs c ON ps.cv_id = c.id
            WHERE p.poste_intitulé = ?
            ORDER BY ps.score DESC
        """, (poste_intitule,))
        rows = c.fetchall()

    if not rows:
        return []

    poste = {'intitule': rows[0][0], 'description': rows[0][1], 'cvs': []}
    for row in rows:
        if row[2] and row[3] is not None:
            poste['cvs'].append({'filename': row[2], 'score': row[3]})

    return [poste]
def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.session_state.user_role = None
    st.session_state.user_id = None
    st.rerun()

# Fonctions existantes modifiées pour utiliser st.session_state

def save_cv(filename, category, score, note, extracted_info, file_data, poste_intitulé, description_poste=None):
    unique_filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    file_path = os.path.join(UPLOAD_DIRECTORY, unique_filename)

    with open(file_path, "wb") as f:
        f.write(file_data)

    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO cvs (filename, category, note, extracted_info, file_path, user_id) VALUES (?, ?, ?, ?, ?, ?)",
            (filename, category, note, extracted_info, file_path, st.session_state.user_id))
        cv_id = c.lastrowid

        if description_poste is not None and poste_intitulé is not None and score is not None:
            # Insérer d'abord dans la table poste
            c.execute(
                "INSERT INTO poste (poste_intitulé, description_poste) VALUES (?, ?)",
                (poste_intitulé, description_poste)
            )
            poste_id = c.lastrowid

            # Puis insérer dans poste_similarite
            c.execute(
                "INSERT INTO poste_similarite (cv_id, poste_id, score, date) VALUES (?, ?, ?, ?)",
                (cv_id, poste_id, score, datetime.now())
            )

        conn.commit()
def get_cvs(category=None, min_score=None):
    query = """
    SELECT cvs.*, poste_similarite.score, poste.poste_intitulé, poste.description_poste
    FROM cvs
    LEFT JOIN poste_similarite ON cvs.id = poste_similarite.cv_id
    LEFT JOIN poste ON poste_similarite.poste_id = poste.id
    """
    params = []
    where_clauses = []

    if category and category != "Tous":
        where_clauses.append("cvs.category = ?")
        params.append(category)

    if min_score is not None:
        where_clauses.append("(poste_similarite.score >= ? OR poste_similarite.score IS NULL)")
        params.append(min_score)

    if st.session_state.user_role != 'admin':
        where_clauses.append("cvs.user_id = ?")
        params.append(st.session_state.user_id)

    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)

    query += " ORDER BY cvs.id DESC"

    with sqlite3.connect('db_cv_thèque.db') as conn:
        c = conn.cursor()
        c.execute(query, params)
        return c.fetchall()

def add_user(username, password, role, nom, prenom):
    if not username or not password:
        raise ValueError("Le nom d'utilisateur et le mot de passe ne peuvent pas être vides.")

    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        with sqlite3.connect('db_cv_thèque.db') as conn:
            c = conn.cursor()
            c.execute("INSERT INTO users (username, password, role, nom, prenom) VALUES (?, ?, ?, ?, ?)",
                      (username, hashed_password.decode('utf-8'), role, nom, prenom))
            conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Ce nom d'utilisateur existe déjà.")
    except Exception as e:
        raise Exception(f"Une erreur s'est produite lors de l'ajout de l'utilisateur : {str(e)}")

def get_user_role(username):
    c.execute("SELECT role FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return result[0] if result else None

def get_user_id(username):
    c.execute("SELECT id FROM users WHERE username = ?", (username,))
    result = c.fetchone()
    return result[0] if result else None
def process_cv(uploaded_file, job_description, nlp):
    try:
        # Extract text based on file type
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()
        if file_extension == '.pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_extension in ['.docx', '.doc']:
            text = extract_text_from_docx(uploaded_file)
        elif file_extension in ['.jpg', '.jpeg', '.png']:
            text = extract_text_from_image(uploaded_file)
        else:
            return f"Type de fichier non pris en charge : {file_extension}", None, None, None, None, None

        if not text.strip():
            return f"Texte vide extrait de {uploaded_file.name}", None, None, None, None, None

        # Translate the CV text to English for processing
        translated_text = translate_to_english(text)

        if not translated_text:
            return f"Échec de la traduction pour {uploaded_file.name}", None, None, None, None, None

        # Categorize the CV
        cv_category = categorize_cv(translated_text)

        # Extract personal information and entities
        doc = nlp(translated_text)
        personal_info = extract_personal_info(translated_text)
        entities = [(translate_to_french(ent.text), translate_to_french(ent.label_)) for ent in doc.ents if
                    ent.label_ not in ["SKILLS", "EMAIL", "PHONE_NUMBER"]]
        for email in personal_info['emails']:
            entities.append((email, "EMAIL"))
        for phone in personal_info['phones']:
            entities.append((phone, "NUMÉRO DE TÉLÉPHONE"))

        # Extract experience details
        experiences = extract_experience_durations(translated_text)

        # Extract skills
        skills_from_list, skills_from_entities = extract_skills(text, doc)
        translated_skills_from_entities = [translate_to_french(skill) for skill in skills_from_entities]
        all_skills = list(set(skills_from_list + translated_skills_from_entities))
        all_skills.sort()

        # Extract education information
        education_info = extract_education(text)

        # Calculate similarity score only if job description is provided
        similarity_score = calculate_similarity(translated_text, job_description) if job_description else None

        extracted_info = {
            'category': cv_category,
            'entities': entities,
            'experiences': experiences,
            'skills': all_skills,
            'education': education_info,
            'raw_text': text  # Add the original text for reference
        }

        # Read file data
        uploaded_file.seek(0)
        file_data = uploaded_file.read()
        return uploaded_file.name, cv_category, similarity_score, extracted_info, file_data, job_description
    except Exception as e:
        st.error(f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}")
        return f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}", None, None, None, None, None
def gestion_utilisateurs():
    st.markdown("""
    <style>
    /* Styles existants */
    .user-row {
        display: flex;
        align-items: center;
        border-bottom: 1px solid #30363d;
        padding: 10px 0;
    }
    .user-row > div {
        padding: 0 10px;
    }
    .user-header {
        font-weight: bold;
        border-bottom: 2px solid #30363d;
        padding-bottom: 10px;
        margin-bottom: 10px;
    }
    .delete-btn {
        background-color: #ff4b4b;
        color: white;
        border: none;
        padding: 5px 10px;
        border-radius: 5px;
        cursor: pointer;
    }

    /* Styles pour réduire l'espace entre les inputs */
    .input-section .stTextInput, .input-section .stSelectbox {
        margin-bottom: -15px;
    }
    .input-section .stTextInput > div > div > input, .input-section .stSelectbox > div > div > div {
        padding: 0.4rem 0.75rem;
    }
    .input-section .stButton {
        margin-top: 1rem;
        display: flex;
        justify-content: center;
    }

    /* Styles pour centrer les titres */
    .centered {
        text-align: center;
    }

    </style>
    """, unsafe_allow_html=True)

    # Titre principal centré en haut
    st.markdown("<h2 class='centered'>Gestion des Utilisateurs</h2>", unsafe_allow_html=True)

    # Section d'ajout d'utilisateur dans un expander
    with st.expander("Ajouter un nouvel utilisateur"):
        # Wrapper div pour la section d'input
        st.markdown('<div class="input-section">', unsafe_allow_html=True)

        new_username = st.text_input("", placeholder="Nom d'utilisateur", key="new_username")
        new_password = st.text_input("", placeholder="Mot de passe", type="password", key="new_password")
        new_role = st.selectbox("", ["user", "admin"], format_func=lambda x: "Rôle" if x == "" else x, key="new_role")
        new_nom = st.text_input("", placeholder="Nom", key="new_nom")
        new_prenom = st.text_input("", placeholder="Prénom", key="new_prenom")

        # Bouton centré
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Ajouter un utilisateur"):
                try:
                    if not new_username or not new_password:
                        st.error("Le nom d'utilisateur et le mot de passe ne peuvent pas être vides.")
                    else:
                        add_user(new_username, new_password, new_role, new_nom, new_prenom)
                        st.success(f"Utilisateur {new_username} ajouté avec succès!")
                except ValueError as ve:
                    st.error(str(ve))
                except Exception as e:
                    st.error(f"Erreur lors de l'ajout de l'utilisateur : {str(e)}")

        # Fermeture du wrapper div
        st.markdown('</div>', unsafe_allow_html=True)

    # Liste des utilisateurs dans un expander
    with st.expander("Liste des utilisateurs"):
        users = get_all_users()
        if users:
            # Table headers
            st.markdown("""
               <div class="user-row user-header">
                   <div style="width: 5%;">ID</div>
                   <div style="width: 20%;">Utilisateur</div>
                   <div style="width: 20%;">Nom</div>
                   <div style="width: 20%;">Prénom</div>
                   <div style="width: 15%;">Rôle</div>
                   <div style="width: 20%;">Action</div>
               </div>
               """, unsafe_allow_html=True)

            # Table rows with Streamlit columns
            for user in users:
                col1, col2, col3, col4, col5, col6 = st.columns([0.5, 2, 2, 2, 1.5, 2])
                with col1:
                    st.write(user[0])
                with col2:
                    st.write(user[1])
                with col3:
                    st.write(user[4])  # Nom
                with col4:
                    st.write(user[5])  # Prénom
                with col5:
                    st.write(user[3])  # Role
                with col6:
                    if st.button("🗑️", key=f"delete_{user[0]}"):
                        try:
                            delete_user(user[0])
                            st.success(f"Utilisateur avec ID {user[0]} supprimé avec succès!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Erreur lors de la suppression de l'utilisateur : {str(e)}")
        else:
            st.info("Aucun utilisateur trouvé.")

def tableau_de_bord():
    cvs = get_cvs()
    columns = ['id', 'filename', 'category', 'note', 'extracted_info', 'file_path', 'user_id', 'score',
               'poste_intitulé', 'description_poste']
    df = pd.DataFrame(cvs, columns=columns)
    df = df.fillna('NULL')

    # Style CSS mis à jour
    st.markdown("""
    <style>
    .metric-container {
        display: flex;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #58606A;
        border-radius: 10px;
        padding: 8px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        width: 120px;
        height: 70px;
        display: flex;
        flex-direction: column;
    }
    .metric-header {
        display: flex;
        align-items: center;
        margin-bottom: 5px;
        
    }
    .metric-icon {
        font-size: 18px;
        color: #5e72e4;
        margin-right: 5px;
    }
    .metric-title {
        font-size: 10px;
        color: #white;
        font-weight: bold;
        
    }
    .metric-value {
        font-size: 16px;
        font-weight: bold;
        text-align: center;
    }
     .chart-container {
        margin-bottom: 10px;
        padding-top: 0;
    }

    </style>
    """, unsafe_allow_html=True)

    # Création des cartes métriques
    metric_html = f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-icon">📊</span>
                <span class="metric-title">Total des CVs</span>
            </div>
            <div class="metric-value">{len(df)}</div>
        </div>
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-icon">📈</span>
                <span class="metric-title">Score moyen</span>
            </div>
            <div class="metric-value">{df[df['score'] != 'NULL']['score'].astype(float).mean():.2f}</div>
        </div>
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-icon">🏷️</span>
                <span class="metric-title">Catégorie principale</span>
            </div>
            <div class="metric-value">{df['category'].value_counts().index[0]}</div>
        </div>
        <div class="metric-card">
            <div class="metric-header">
                <span class="metric-icon">⭐</span>
                <span class="metric-title">Note moyenne</span>
            </div>
            <div class="metric-value">{df['note'].astype(float).mean():.2f}</div>
        </div>
    </div>
    """

    st.markdown(metric_html, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["Catégories", "Notes et Scores", "Évolution et Compétences"])
    col1, col2 = st.columns(2)

    with tab1:
        # Chart 1: Distribution des catégories de CV
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="chart-title">Distribution des catégories de CV</h3>', unsafe_allow_html=True)
            category_counts = df['category'].value_counts().reset_index()
            category_counts.columns = ['category', 'count']

            fig = px.pie(category_counts, values='count', names='category',
                         color_discrete_sequence=px.colors.qualitative.Set3,
                         hover_data=['count'])

            fig.update_traces(textposition='inside', textinfo='percent+label')
            fig.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(l=20, r=20, t=20, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                height=350,
            )

            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    with tab2:


            # Chart 2: Distribution des scores de similarité
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="chart-title">Distribution des scores de similarité</h3>', unsafe_allow_html=True)
                score_df = df[df['score'] != 'NULL'].copy()
                score_df['score'] = pd.to_numeric(score_df['score'], errors='coerce')
                score_df = score_df.dropna(subset=['score'])
                if not score_df.empty:
                    score_df['score_bin'] = pd.cut(score_df['score'], bins=10,
                                                   labels=[f"{i * 10}-{(i + 1) * 10}" for i in range(10)])
                    score_counts = score_df['score_bin'].value_counts().sort_index()
                    st.bar_chart(score_counts, height=300)
                else:
                    st.info("Pas de données de score disponibles pour créer l'histogramme.")
                st.markdown('</div>', unsafe_allow_html=True)


            # Chart 3: Notes moyennes par catégorie
            with st.container():
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.markdown('<h3 class="chart-title">Notes moyennes par catégorie</h3>', unsafe_allow_html=True)
                df['note'] = pd.to_numeric(df['note'], errors='coerce')
                avg_notes = df.groupby('category')['note'].mean().sort_values(ascending=False)
                if not avg_notes.empty:
                    st.bar_chart(avg_notes, height=300)
                else:
                    st.info("Pas de données de note disponibles pour calculer les moyennes par catégorie.")
                st.markdown('</div>', unsafe_allow_html=True)
    with tab3:
        # Chart 4: Évolution des soumissions de CV dans le temps
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="chart-title">Évolution des soumissions de CV dans le temps</h3>', unsafe_allow_html=True)

            df['submission_date'] = pd.to_datetime(df['file_path'].str.extract(r'(\d{14})')[0], format='%Y%m%d%H%M%S',
                                                   errors='coerce')

            submissions_over_time = df.groupby(df['submission_date'].dt.date).size().reset_index(name='count')
            submissions_over_time = submissions_over_time.sort_values('submission_date')

            if not submissions_over_time.empty:
                fig = px.line(submissions_over_time, x='submission_date', y='count')
                fig.update_layout(xaxis_title="Date", yaxis_title="Nombre de soumissions", height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Pas de données de soumission disponibles pour créer le graphique d'évolution.")
            st.markdown('</div>', unsafe_allow_html=True)

        # Chart 5: Top 10 des compétences les plus fréquentes
        with st.container():
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.markdown('<h3 class="chart-title">Top 10 des compétences les plus fréquentes</h3>', unsafe_allow_html=True)
            all_skills = []
            for info in df['extracted_info']:
                try:
                    extracted_info = eval(info)
                    all_skills.extend(extracted_info.get('skills', []))
                except:
                    pass
            if all_skills:
                skill_counts = pd.Series(all_skills).value_counts().head(10)
                st.bar_chart(skill_counts, height=300)
            else:
                st.info("Pas de données de compétences disponibles pour créer le top 10.")
            st.markdown('</div>', unsafe_allow_html=True)

def create_pdf_display_html(pdf_data):
    """Crée le HTML pour afficher un PDF en base64"""
    b64_pdf = base64.b64encode(pdf_data).decode('utf-8')
    return f"""
        <iframe src="data:application/pdf;base64,{b64_pdf}" 
                width="100%" 
                height="800px" 
                type="application/pdf">
        </iframe>
    """

def create_doc_preview(doc_data):
    """Convertit un fichier DOC/DOCX en texte pour l'aperçu"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as tmp_file:
        tmp_file.write(doc_data)
        tmp_file.flush()

        try:
            import textract
            text = textract.process(tmp_file.name).decode('utf-8')
        except:
            text = "Impossible de lire le contenu du document. Veuillez télécharger le fichier."
        finally:
            os.unlink(tmp_file.name)
    return text

def afficher_fichier_modal(file_data, filename):
    """Affiche le fichier dans une section dédiée"""
    file_extension = filename.split('.')[-1].lower()

    # Utiliser une clé unique pour chaque fichier
    show_preview_key = f"show_preview_{filename}"

    # Bouton pour afficher/masquer l'aperçu
    if st.button(f"👁️ Visualiser {filename}", key=f"view_button_{filename}"):
        st.session_state[show_preview_key] = not st.session_state.get(show_preview_key, False)

    # Afficher le contenu si le bouton a été cliqué
    if st.session_state.get(show_preview_key, False):
        st.markdown("### 📄 Aperçu du document")

        if file_extension in ['jpg', 'jpeg', 'png']:
            try:
                image = Image.open(io.BytesIO(file_data))
                st.image(image, caption=filename, use_column_width=True)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage de l'image: {str(e)}")

        elif file_extension == 'pdf':
            try:
                st.markdown(create_pdf_display_html(file_data), unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du PDF: {str(e)}")

        elif file_extension in ['doc', 'docx']:
            try:
                preview_text = create_doc_preview(file_data, filename)
                # Ajouter une clé unique pour le text_area
                st.text_area(
                    "Contenu du document",
                    value=preview_text,
                    height=400,
                    key=f"doc_preview_{filename}"
                )
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du document: {str(e)}")

        # Bouton de téléchargement placé sous l'affichage
        st.download_button(
            "⬇️ Télécharger le fichier",
            file_data,
            filename,
            help="Télécharger la version originale du fichier",
            key=f"download_{filename}"
        )

        # Ajout d'un séparateur pour une meilleure organisation visuelle
        st.markdown("---")

def create_doc_preview(doc_data, filename):
    """Convertit un fichier DOC/DOCX en texte pour l'aperçu"""
    try:
        import textract
        # Créer un nom de fichier temporaire unique basé sur le nom du fichier original
        temp_filename = f"temp_{filename}_{os.getpid()}"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)

        try:
            with open(temp_path, 'wb') as tmp_file:
                tmp_file.write(doc_data)

            text = textract.process(temp_path).decode('utf-8')
            return text

        except Exception as e:
            return f"Erreur lors de la lecture du document: {str(e)}"

        finally:
            # Essayer de supprimer le fichier temporaire
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except:
                pass  # Ignorer les erreurs de suppression

    except ImportError:
        return "Module textract non installé. Impossible de lire le contenu du document."


def analyser_cvs():
    st.header("Analyser de nouveaux CVs")
    nlp = load_model()
    poste_intitulé = st.text_input("Entrez l'intitulé du poste")
    job_description = st.text_area("Entrez la description du poste", height=200, key="job_description_input")
    uploaded_files = st.file_uploader("Choisissez des fichiers CV",
                                      type=["pdf", "docx", "doc", "jpg", "jpeg", "png"],
                                      accept_multiple_files=True)
    if uploaded_files:
        results = []

        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_cv = {executor.submit(process_cv, file, job_description, nlp): file for file in uploaded_files}
            for future in concurrent.futures.as_completed(future_to_cv):
                results.append(future.result())

        # Trier les résultats par score si une description de poste est fournie
        if job_description:
            valid_results = [r for r in results if isinstance(r, tuple) and len(r) == 6 and r[2] is not None]
            sorted_results = sorted(valid_results, key=lambda x: x[2], reverse=True)

            with st.expander("Voir le classement des candidats"):
                for rank, (filename, _, score, _, _, _) in enumerate(sorted_results, 1):
                    st.markdown(f"{rank}. {filename} - Score: {score:.2f}%")

        for i, result in enumerate(results):
            if isinstance(result, tuple) and len(result) == 6:
                filename, category, score, extracted_info, file_data, job_desc = result
                with st.expander(f"Résultats pour {filename}"):
                    if category is not None and extracted_info is not None:
                        # Traduire la catégorie en français
                        french_category = translate_to_french(category)
                        new_category = st.text_input(
                            f"Catégorie du CV (modifiable) - {filename}",
                            value=french_category,
                            key=f"category_{filename}"
                        )

                        if job_description and score is not None:
                            st.success(f"Score de similarité avec la description du poste : {score:.2f}%")

                        # Afficher le fichier avec le nouveau système
                        afficher_fichier_modal(file_data, filename)

                        #st.subheader("Texte extrait du CV")
                        #st.text_area(
                            #"Texte brut",
                            #value=extracted_info.get('raw_text', ''),
                            #height=200,
                            #key=f"raw_text_{filename}"
                        #)

                        st.subheader("Entités Reconnues")
                        st.table(pd.DataFrame(extracted_info['entities'], columns=['Texte', 'Type d\'Entité']))

                        st.subheader("Expériences extraites")
                        if extracted_info['experiences']:
                            for j, exp in enumerate(extracted_info['experiences']):
                                st.markdown(f"""
                                    **Période:** {exp.get('date_range', '')}<br>
                                    **Expérience Info:** {translate_to_french(exp.get('job_info', ''))}<br>
                                    {translate_to_french(exp.get('company_info', ''))}<br>
                                    **Durée:** {translate_to_french(exp.get('duration', ''))}
                                    <hr>
                                    """, unsafe_allow_html=True)
                        else:
                            st.info("Aucune expérience n'a été extraite de ce CV")

                        st.subheader("Compétences extraites")
                        display_skills(extracted_info['skills'])

                        cv_note = st.number_input(
                            f"Ajouter une note pour le CV (entre 0 et 100) - {filename}",
                            min_value=0,
                            max_value=100,
                            value=50,
                            step=1,
                            key=f"note_{filename}"
                        )

                        if st.button(f"Enregistrer {filename}", key=f"save_{filename}"):
                            try:
                                save_cv(filename, new_category, score, cv_note, str(extracted_info), file_data,
                                        poste_intitulé, job_desc)
                                st.success(f"{filename} enregistré avec succès!")
                            except Exception as e:
                                st.error(f"Erreur lors de l'enregistrement du CV : {str(e)}")
                    else:
                        st.error(filename)
            else:
                st.error(f"Erreur lors du traitement d'un fichier : {result}")

def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None


def home_page():
    st.markdown("""
    <style>
    .big-font {
        font-size:24px !important;
        font-weight: bold;
        text-align: center;
    }
    .medium-font {
        font-size:14px !important;
        text-align: center;
        color: #636363;
    }
    .feature-title {
        font-size:18px !important;
        font-weight: bold;
        text-align: center;
    }
    .feature-desc {
        font-size:14px !important;
        color: #636363;
        text-align: center;
    }
    .feature-icon {
        font-size:64px !important;
        text-align: center;
        color: #4B0082;
    }
    .start-button {
        background-color: #4B0082;
        color: white;
        padding: 0.5rem 2rem;
        border-radius: 4px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        transition: background-color 0.3s;
        display: block;
        margin: 0 auto;
        text-align: center;
    }
    .start-button:hover {
        background-color: #3a006c;
    }
    </style>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    features = [
        {
            "icon": "fas fa-file-alt",
            "title": "Analyse de CV",
            "desc": "Analysez et catégorisez automatiquement les CV pour une gestion efficace des candidatures.",
            "lottie": "https://assets5.lottiefiles.com/packages/lf20_pwohahvd.json"
        },
        {
            "icon": "fas fa-chart-bar",
            "title": "Tableau de Bord",
            "desc": "Visualisez des statistiques détaillées et des insights sur votre base de CV.",
            "lottie": "https://assets3.lottiefiles.com/packages/lf20_qp1q7mct.json"
        }
    ]

    for col, feature in zip([col1, col2], features):
        with col:
            lottie_animation = load_lottieurl(feature['lottie'])
            if lottie_animation:
                st_lottie(lottie_animation, height=200)
            else:
                st.markdown(f'<i class="feature-icon {feature["icon"]}"></i>', unsafe_allow_html=True)
            st.markdown(f'<p class="feature-title">{feature["title"]}</p>', unsafe_allow_html=True)
            st.markdown(f'<p class="feature-desc">{feature["desc"]}</p>', unsafe_allow_html=True)

    st.markdown('<br>', unsafe_allow_html=True)

    # Centrer le bouton avec trois colonnes
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button('Commencer l\'analyse', key='start_analysis', use_container_width=True):
            # Mettre à jour le choix dans la session state
            st.session_state.choice = "Analyser CVs"
            # Forcer le rechargement de la page pour appliquer le changement
            st.rerun()

    # Add Font Awesome CDN
    st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">
    """, unsafe_allow_html=True)

def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


def main():
    #db = Database()
    init_session_state()
    # Configuration du thème sombre
    st.set_page_config(page_title="Analyseur de CV", layout="wide", initial_sidebar_state="expanded")
    st.markdown("""
    <style>
    .main .block-container {
    padding-top: 1rem;
    padding-bottom: 0rem;
    }
    .main-content h1 {
    margin-top: -10px;
    }
    .main-container {
    max-width: 900px;
    margin: 0 auto;
    padding: 0;
    box-sizing: border-box;
    }
    .chart-container {
     margin-top: -1rem; 
    }
    .chart-title {
    text-align: center;
    font-weight: bold;
    margin-bottom: 10px;
    color: #333;
    }
    .stPlotlyChart {
    width: 100%;
    height: 400px;
    }
    @media (max-width: 768px) {
    .main-container {
    padding: 0 10px;
    }
    }
    .lottie-container {
    width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    choice = styled_sidebar()

    # Create three columns
    left_spacer, content, right_spacer = st.columns([1, 6, 1])

    # Use the middle column for the main content
    with content:
        if not st.session_state.authenticated:


            # Animation Lottie
            lottie_url = "https://assets9.lottiefiles.com/packages/lf20_jcikwtux.json"  # Remplacez par l'URL de votre animation
            lottie_json = load_lottieurl(lottie_url)
            if lottie_json:
                st.markdown('<div class="lottie-container">', unsafe_allow_html=True)
                st_lottie(lottie_json, key="welcome_animation", height="100%", width="100%")
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.error("Impossible de charger l'animation. Veuillez vérifier votre connexion internet.")
            #st.title("")
            #st.write("Veuillez vous connecter pour accéder à l'application.")
        else:
            with st.container():
                st.markdown('<div class="main-container">', unsafe_allow_html=True)

                if st.session_state.choice == "Accueil":
                    home_page()
                elif st.session_state.choice == "Analyser CVs":
                    analyser_cvs()
                elif st.session_state.choice == "Tableau de Bord":
                    tableau_de_bord()
                elif st.session_state.choice == "Filtrage CV":
                    filtrage_cv()
                elif st.session_state.choice == "Filtrage Postes":
                    filtrage_postes()
                elif st.session_state.choice == "Gestion des Utilisateurs" and st.session_state.user_role == 'admin':
                    gestion_utilisateurs()
                st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()