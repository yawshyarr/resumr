from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import spacy
import re

# ---------- Load SpaCy Model ----------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise RuntimeError(
        "SpaCy model 'en_core_web_sm' not found. Run:\n"
        "   python -m spacy download en_core_web_sm"
    )

# ---------- Preprocessing ----------
def preprocess_text(text):
    """Enhanced text preprocessing"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s+#]', ' ', text)  # keep alphanumeric and '#'
    text = re.sub(r'\s+', ' ', text)  # remove multiple spaces
    return text.strip()

# ---------- Similarity ----------
def compute_similarity(jd_text, resume_texts, filenames):
    """Compute similarity between job description and resumes."""
    jd_processed = preprocess_text(jd_text)
    resume_processed = [preprocess_text(t) for t in resume_texts]

    documents = [jd_processed] + resume_processed

    if not any(documents):
        return pd.DataFrame({"Resume": filenames, "Match %": [0] * len(filenames)})

    tfidf = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        max_features=500,
        min_df=1,
        max_df=0.95
    )

    try:
        tfidf_matrix = tfidf.fit_transform(documents)
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        # Keyword-based scoring
        jd_keywords = extract_keywords(jd_text)
        keyword_scores = []
        for resume_text in resume_texts:
            resume_keywords = extract_keywords(resume_text)
            common_keywords = jd_keywords & resume_keywords
            keyword_match = len(common_keywords) / len(jd_keywords) if jd_keywords else 0
            keyword_scores.append(keyword_match)

        # Weighted score
        final_scores = [
            (0.7 * cos + 0.3 * kw)
            for cos, kw in zip(cosine_similarities, keyword_scores)
        ]

    except Exception as e:
        print(f"[ERROR] TF-IDF failed: {e}")
        final_scores = [0.5] * len(resume_texts)

    df = pd.DataFrame({
        "Resume": filenames,
        "Match %": [round(score * 100, 2) for score in final_scores]
    })

    return df.sort_values(by="Match %", ascending=False).reset_index(drop=True)

# ---------- Keyword Extraction ----------
def extract_keywords(text):
    """Extract keywords including technical terms."""
    if not text:
        return set()

    doc = nlp(text.lower())
    keywords = set()

    # Tech terms regex
    tech_terms = re.findall(
        r'\b(python|java|javascript|react|angular|vue|django|flask|sql|mysql|postgresql|mongodb|'
        r'aws|azure|gcp|docker|kubernetes|git|api|rest|graphql|machine learning|deep learning|ai|ml|'
        r'data science|analytics|agile|scrum|devops|ci/cd|html|css|bootstrap|node|express|spring|'
        r'tensorflow|pytorch|pandas|numpy|sklearn)\b',
        text.lower()
    )
    keywords.update(map(str.lower, tech_terms))

    # Nouns, proper nouns, adjectives
    for token in doc:
        if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and
                not token.is_stop and
                len(token.text) > 2 and
                token.text.isalnum()):
            keywords.add(token.lemma_.lower())

    # Short noun phrases
    for chunk in doc.noun_chunks:
        if len(chunk.text.split()) <= 3:
            keywords.add(chunk.text.lower().strip())

    return keywords

# ---------- Missing Keywords ----------
def missing_keywords(jd_text, resume_text):
    """Find keywords from JD missing in resume."""
    jd_keywords = extract_keywords(jd_text)
    resume_keywords = extract_keywords(resume_text)

    missing = jd_keywords - resume_keywords
    filtered = {
        kw for kw in missing if len(kw) > 2 and not kw.isdigit()
    }

    # Sort by length (longer = more specific)
    sorted_missing = sorted(filtered, key=len, reverse=True)

    return set(sorted_missing[:15])

# ---------- Skill Categorization ----------
def get_skill_categories(text):
    """Categorize skills found in text."""
    skills = {
        "Programming Languages": [],
        "Frameworks": [],
        "Databases": [],
        "Tools & Technologies": [],
        "Soft Skills": []
    }

    text_lower = text.lower()

    programming_languages = ["python", "java", "javascript", "c++", "c#", "ruby", "go", "rust",
                             "php", "swift", "kotlin", "r", "matlab", "typescript"]
    frameworks = ["react", "angular", "vue", "django", "flask", "express", "spring", "laravel",
                  "rails", ".net", "tensorflow", "pytorch", "keras"]
    databases = ["mysql", "postgresql", "mongodb", "redis", "oracle", "sql server",
                 "dynamodb", "cassandra", "elasticsearch"]
    tools = ["docker", "kubernetes", "git", "jenkins", "aws", "azure", "gcp", "linux",
             "windows", "jira", "confluence"]
    soft_skills = ["leadership", "communication", "teamwork", "problem solving", "analytical",
                   "creative", "management", "collaboration"]

    for lang in programming_languages:
        if re.search(r"\b" + re.escape(lang) + r"\b", text_lower):
            skills["Programming Languages"].append(lang.capitalize())

    for fw in frameworks:
        if re.search(r"\b" + re.escape(fw) + r"\b", text_lower):
            skills["Frameworks"].append(fw.capitalize())

    for db in databases:
        if re.search(r"\b" + re.escape(db) + r"\b", text_lower):
            skills["Databases"].append(db.upper() if len(db) <= 5 else db.capitalize())

    for tool in tools:
        if re.search(r"\b" + re.escape(tool) + r"\b", text_lower):
            skills["Tools & Technologies"].append(tool.upper() if len(tool) <= 3 else tool.capitalize())

    for skill in soft_skills:
        if skill in text_lower:
            skills["Soft Skills"].append(skill.title())

    return skills
