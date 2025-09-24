from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import tempfile
import traceback

# Import your analyzer modules
from analyzer import compute_similarity, missing_keywords, extract_keywords
from resume_extractor import extract_resume_text

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = 'your-secret-key-here'

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

# ---------- Utility ----------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------- Routes ----------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        jd_text = request.form.get('job_description', '').strip()
        if not jd_text:
            return jsonify({'error': 'Please provide a job description'}), 400

        if 'resumes' not in request.files:
            return jsonify({'error': 'No resumes uploaded'}), 400

        files = request.files.getlist('resumes')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No resumes selected'}), 400

        resume_texts, filenames = [], []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as tmp_file:
                    file.save(tmp_file.name)
                    text = extract_resume_text(tmp_file.name, filename)
                    os.unlink(tmp_file.name)  # clean up

                if text:
                    resume_texts.append(text)
                    filenames.append(filename)

        if not resume_texts:
            return jsonify({'error': 'Could not extract text from any resume'}), 400

        result_df = compute_similarity(jd_text, resume_texts, filenames)

        missing_keywords_data = []
        for i, name in enumerate(filenames):
            missing = missing_keywords(jd_text, resume_texts[i])
            missing_keywords_data.append({
                'filename': name,
                'missing': list(missing)
            })

        jd_keywords = list(extract_keywords(jd_text))[:20]

        response = {
            'rankings': result_df.to_dict(orient='records'),
            'missing_keywords': missing_keywords_data,
            'jd_keywords': jd_keywords,
            'total_resumes': len(filenames)
        }

        return jsonify(response), 200

    except Exception as e:
        import traceback
        print("❌ ERROR in /analyze:", traceback.format_exc())
        return jsonify({'error': str(e)}), 500
