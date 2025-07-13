from flask import Flask, request, render_template_string
import joblib

app = Flask(__name__)

def load_model():
    """Load model when needed"""
    try:
        return joblib.load('Models/fake_news_rf_pipeline.pkl')
    except Exception as e:
        return None, str(e)

def analyze_headline(headline, pipeline):
    """Improved analysis with confidence thresholds"""
    try:
        prediction = pipeline.predict([headline])[0]
        probabilities = pipeline.predict_proba([headline])[0]
        confidence = max(probabilities)
        
        # Get individual probabilities
        fake_prob = probabilities[0]  # Probability of being fake
        real_prob = probabilities[1]  # Probability of being real
        
        # Improved logic with confidence thresholds
        if confidence < 0.65:  # Low confidence
            if real_prob > fake_prob:
                result_type = "uncertain-real"
                message = "🤔 PROBABLY REAL (Low Confidence)"
                explanation = f"The model thinks this is real news but isn't very confident. Real: {real_prob:.1%} vs Fake: {fake_prob:.1%}"
            else:
                result_type = "uncertain-fake"
                message = "🤔 PROBABLY FAKE (Low Confidence)"
                explanation = f"The model thinks this is fake news but isn't very confident. Fake: {fake_prob:.1%} vs Real: {real_prob:.1%}"
        else:  # High confidence
            if prediction == 1:
                result_type = "real"
                message = "✅ REAL NEWS (High Confidence)"
                explanation = f"The model is confident this is real news. Real: {real_prob:.1%} vs Fake: {fake_prob:.1%}"
            else:
                result_type = "fake"
                message = "🚨 FAKE NEWS (High Confidence)"
                explanation = f"The model is confident this is fake news. Fake: {fake_prob:.1%} vs Real: {real_prob:.1%}"
        
        return {
            'type': result_type,
            'message': message,
            'confidence': f"{confidence:.1%}",
            'explanation': explanation,
            'fake_prob': f"{fake_prob:.1%}",
            'real_prob': f"{real_prob:.1%}"
        }
        
    except Exception as e:
        return {
            'type': 'error',
            'message': f'❌ Prediction Error: {str(e)}',
            'confidence': None,
            'explanation': None,
            'fake_prob': None,
            'real_prob': None
        }

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Fake News Detection System</title>
    <style>
        body { font-family: Arial; margin: 40px; background-color: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .result { padding: 20px; margin: 20px 0; border-radius: 8px; }
        .real { background-color: #d4edda; color: #155724; border-left: 5px solid #28a745; }
        .fake { background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
        .uncertain-real { background-color: #fff3cd; color: #856404; border-left: 5px solid #ffc107; }
        .uncertain-fake { background-color: #f8d7da; color: #721c24; border-left: 5px solid #fd7e14; }
        .error { background-color: #f8d7da; color: #721c24; border-left: 5px solid #dc3545; }
        textarea { width: 100%; height: 120px; padding: 10px; border: 2px solid #ddd; border-radius: 5px; font-size: 14px; }
        .btn { padding: 12px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; cursor: pointer; font-size: 14px; }
        .btn:hover { background: #0056b3; }
        .btn-example { background: #6c757d; }
        .btn-example:hover { background: #545b62; }
        .stats { display: flex; justify-content: space-around; margin: 15px 0; }
        .stat { text-align: center; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .explanation { margin-top: 10px; font-size: 14px; line-height: 1.4; }
        h1 { color: #333; text-align: center; }
        .examples-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📰 Enhanced Fake News Detection System</h1>
        
        <form method="POST">
            <p><strong>Enter a news headline:</strong></p>
            <textarea name="headline" placeholder="Enter a complete news headline here... (The model works better with longer, detailed headlines)">{{ headline or '' }}</textarea>
            <br><br>
            <button type="submit" class="btn">🔍 Analyze Headline</button>
        </form>
        
        {% if result %}
        <div class="result {{ result.type }}">
            <h3>{{ result.message }}</h3>
            {% if result.explanation %}
            <div class="explanation">{{ result.explanation }}</div>
            {% endif %}
            
            {% if result.fake_prob and result.real_prob %}
            <div class="stats">
                <div class="stat">
                    <strong>Fake Probability</strong><br>
                    {{ result.fake_prob }}
                </div>
                <div class="stat">
                    <strong>Real Probability</strong><br>
                    {{ result.real_prob }}
                </div>
                <div class="stat">
                    <strong>Overall Confidence</strong><br>
                    {{ result.confidence }}
                </div>
            </div>
            {% endif %}
        </div>
        {% endif %}
        
        <hr>
        <h3>📚 Test Examples:</h3>
        <div class="examples-grid">
            <form method="POST">
                <input type="hidden" name="headline" value="Federal judge partially lifts Trump's latest refugee restrictions">
                <button type="submit" class="btn btn-example">📰 Real News Example</button>
            </form>
            
            <form method="POST">
                <input type="hidden" name="headline" value="Dwayne 'The Rock' Johnson Dies at 47 After a Terrible Stunt Attempt Failed">
                <button type="submit" class="btn btn-example">🚨 Fake News Example</button>
            </form>
            
            <form method="POST">
                <input type="hidden" name="headline" value="Scientists discover breakthrough cancer treatment in clinical trials">
                <button type="submit" class="btn btn-example">🔬 Science Example</button>
            </form>
            
            <form method="POST">
                <input type="hidden" name="headline" value="Aliens demand meeting with world leaders after landing in major cities">
                <button type="submit" class="btn btn-example">👽 Obviously Fake Example</button>
            </form>
            
            <form method="POST">
                <input type="hidden" name="headline" value="President announces new economic stimulus package to support small businesses">
                <button type="submit" class="btn btn-example">🏛️ Political Example</button>
            </form>
            
            <form method="POST">
                <input type="hidden" name="headline" value="Local man claims he can speak to dolphins after eating mysterious seaweed">
                <button type="submit" class="btn btn-example">🐬 Weird Example</button>
            </form>
        </div>
        
        <hr>
        <div style="background: #e9ecef; padding: 15px; border-radius: 5px; margin-top: 20px;">
            <h4>🔍 How This Works:</h4>
            <ul>
                <li><strong>High Confidence (65%+):</strong> Model is fairly certain about its prediction</li>
                <li><strong>Low Confidence (&lt;65%):</strong> Model is uncertain - take results with skepticism</li>
                <li><strong>Better Results:</strong> Use longer, complete headlines rather than short phrases</li>
                <li><strong>Training Data:</strong> Model learned from real/fake news articles, may not recognize all types of misinformation</li>
            </ul>
        </div>
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    headline = None
    
    if request.method == 'POST':
        headline = request.form.get('headline', '').strip()
        
        if headline:
            pipeline = load_model()
            
            if isinstance(pipeline, tuple):  # Error loading
                result = {
                    'type': 'error',
                    'message': f'❌ Model Error: {pipeline[1]}',
                    'confidence': None,
                    'explanation': None,
                    'fake_prob': None,
                    'real_prob': None
                }
            else:
                result = analyze_headline(headline, pipeline)
    
    return render_template_string(HTML_TEMPLATE, result=result, headline=headline)

if __name__ == '__main__':
    print("🚀 Starting Enhanced Fake News Detection System...")
    print("🔍 Features: Confidence thresholds, detailed analysis, more examples")
    print("🌐 Open: http://localhost:5000")
    app.run(debug=True, port=5000)