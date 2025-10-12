from flask import Flask, render_template, request, redirect, url_for, flash, session
import os
import pandas as pd
from werkzeug.utils import secure_filename
from ml.pre_processing import preprocessing_data

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'data'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create data directory if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    """Homepage with file upload form"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload"""
    if 'file' not in request.files:
        flash('No file selected')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = 'x_test.csv'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Store filename in session for analysis page
        session['uploaded_file'] = filename
        
        # Basic file analysis
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            else:
                df = pd.read_excel(filepath)
            
            # Store basic info in session
            session['file_info'] = {
                'filename': filename,
                'rows': len(df),
                'columns': len(df.columns),
                'column_names': list(df.columns)
            }
            
            # Preprocess data
            processed_data = preprocessing_data(df , drop_cols=["latitude" , "longitude"  , "county" , "state" , "cust_orig_date" , "date_of_birth" , "acct_suspd_date" ] , categorical_encoder='onehot' , path= app.config['UPLOAD_FOLDER'])

            return redirect(url_for('predictions'))
            
        except Exception as e:
            flash(f'Error processing file: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload CSV or Excel files only.')
        return redirect(url_for('index'))

@app.route('/predictions')
def predictions():
    """AI predictions page with analysis results"""
    if 'file_info' not in session:
        flash('Please upload a file first')
        return redirect(url_for('index'))
    
    file_info = session.get('file_info')
    
    # Placeholder predictions data
    placeholder_predictions = {
        'accuracy_score': 92.5,
        'model_type': 'Random Forest Classifier',
        'feature_importance': [
            {'feature': 'Feature_1', 'importance': 0.35},
            {'feature': 'Feature_2', 'importance': 0.28},
            {'feature': 'Feature_3', 'importance': 0.22},
            {'feature': 'Feature_4', 'importance': 0.15}
        ],
        'predictions_summary': {
            'total_predictions': file_info['rows'] if file_info else 100,
            'positive_predictions': 67,
            'negative_predictions': 33
        },
        'model_metrics': {
            'precision': 0.91,
            'recall': 0.89,
            'f1_score': 0.90
        }
    }
    
    return render_template('predictions.html', 
                         file_info=file_info, 
                         predictions=placeholder_predictions)

@app.route('/reset')
def reset():
    """Reset session and go back to upload page"""
    session.clear()
    flash('Session reset. You can upload a new file.')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)