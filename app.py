from flask import Flask, request, jsonify
import pickle

# Load the trained model
with open("random_forest_sentiment.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the trained TF-IDF vectorizer
with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure the request has JSON data
        if not request.is_json:
            return jsonify({"error": "Invalid request format. Content-Type must be application/json"}), 415

        # Debugging: Print the received request data
        data = request.get_json()
        print("Received request data:", data)

        # Validate if 'review' key exists
        if not data or "review" not in data or not data["review"].strip():
            return jsonify({"error": "Review text is required"}), 400
        
        review = data["review"]

        # Transform input using the trained vectorizer
        review_transformed = vectorizer.transform([review])

        # Predict sentiment
        prediction = model.predict(review_transformed)[0]

        # Return response
        return jsonify({"review": review, "sentiment": "positive" if prediction == 1 else "negative"})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
