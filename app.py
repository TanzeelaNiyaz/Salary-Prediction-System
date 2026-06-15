from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("gbm_model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    job_title = request.form["job_title"].strip()
    education_level = request.form["education_level"].strip()
    industry = request.form["industry"].strip()
    company_size = request.form["company_size"].strip()
    location = request.form["location"].strip()
    remote_work = request.form["remote_work"].strip()



    experience_years = float(request.form["experience_years"])
    skills_count = float(request.form["skills_count"])
    certifications = float(request.form["certifications"])

    cat_data = [[
        job_title,
        education_level,
        industry,
        company_size,
        location,
        remote_work
    ]]

    try:

        cat_encoded = encoder.transform(cat_data)
    except Exception as e:

        return render_template(
        "index.html",
        prediction_text=f"Invalid category entered: {e}"
    )

    final_data = np.hstack([
        [
            experience_years,
            skills_count,
            certifications
        ],
        cat_encoded[0]
    ])

    prediction = model.predict(final_data.reshape(1, -1))

    return render_template(
        "index.html",
        prediction_text=f"Predicted Salary: ${prediction[0]:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)