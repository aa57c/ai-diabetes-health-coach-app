# 🩺 AI-Powered Diabetes Health Coach

An AI-driven web application that predicts diabetes risk and provides personalized health recommendations using structured data and CGM (Continuous Glucose Monitoring) trends. This app combines Machine Learning and Deep Learning with Large Language Models to assist users in managing their health proactively.

---
## Demo
This demo was recorded during Fall Semester 2024 at University of Missouri-Kansas City for the UMKC Rack-A-Roo Competition. I want to give thanks to Professor Yugyung Lee and my teammates Sai, Venkat, and Sreevardhan for helping me during the development process: https://youtu.be/ilnozF_kpas?si=1PN9cMrSxw3REIZA

## Deployed App
The app can be found at this address: https://diabetes-health-coach.streamlit.app/

## 🚀 App Features

- **🧠 Diabetes Prediction** using structured health data and CGM input
- **📈 Personalized Health Recommendations** generated using LLMs
- **💾 User Data Storage** via MongoDB
- **📦 Models stored on Amazon S3**
- **🖥️ LLM API hosted on Amazon EC2**
- **🌐 Deployed on Streamlit Community Cloud**

---

## 🛠️ Tech Stack

| Component          | Technology                        |
|--------------------|-----------------------------------|
| Frontend           | [Streamlit](https://streamlit.io) |
| ML Models          | XGBoost, LSTM                     |
| LLM Server         | Hosted on **Amazon EC2**          |
| Model Storage      | **Amazon S3**                     |
| Backend Database   | **MongoDB Atlas** (Cloud-hosted)  |
| Deployment         | **Streamlit Community Cloud**     |

---

## 📂 Project Structure

```bash
ai-diabetes-health-coach-app/
│
├── models/                # Placeholder if local testing is needed
├── utils/                 # Helper functions
├── api/                   # LLM API endpoint (deployed on EC2)
├── app.py                 # Main Streamlit app
├── requirements.txt       # Python dependencies
└── README.md              # You're here!
```

---

## ⚙️ How It Works

1. **User Input**:
   - Health and lifestyle data collected via forms
   - Optional: Upload CGM data (for deeper insights)

2. **Prediction**:
   - ML models (XGBoost for structured data, LSTM for CGM) loaded from S3
   - Combined prediction generated

3. **LLM Recommendation**:
   - User's data and prediction sent to an EC2-hosted API
   - LLM generates personalized advice and educational info
   - WARNING: Because of resource contraints, the models are run on EC2 using CPU not GPU. So the recommendations might take some time to load. This feature will further be improved in due course.

4. **Storage**:
   - All inputs, predictions, and recommendations stored in MongoDB

---

## 🧪 Local Setup (for testing)

> Note: For local testing, you must mock AWS S3 and MongoDB connections.

```bash
# Clone the repo
git clone https://github.com/your-username/ai-diabetes-health-coach-app.git
cd ai-diabetes-health-coach-app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## ☁️ Deployment Overview

- **Models**: Uploaded to an S3 bucket (`s3://your-model-bucket`)
- **LLM Server**: Flask/FastAPI server hosted on EC2 (`http://ec2-your-ip:port`)
- **Database**: MongoDB Atlas handles input logs and user interaction history
- **Frontend**: Hosted on [Streamlit Community Cloud](https://streamlit.io/cloud)

---

## 🔐 Security & Privacy

- No personal identifiers are stored.
- All user interactions are anonymized and securely stored.
- Communication with the LLM API and database is encrypted.

---

## 🧩 Future Improvements

- Enhanced security for sensitive healthcare data
- Retraining pipeline for continuous model improvement
- Expanded health metrics and condition support
- Feedback loop for LLM recommendation accuracy

---

## 📬 Contact

For questions or feedback, please reach out at ashna.ali.prof@gmail.com or open an issue.


    

