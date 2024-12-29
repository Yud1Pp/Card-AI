# KardAI: Cardiovascular Diagnosis

KardAI is a website designed to assess the risk of cardiovascular diseases (CVD) and provide a conversational chatbot for education about heart health.

---

## What is Cardiovascular Disease?
Cardiovascular diseases are a group of disorders affecting the heart and blood vessels. These include:
- **Coronary Heart Disease**
- **Peripheral Artery Disease**
- **Heart Failure**
- **Congenital Heart Defects**
- **Heart Valve Disease**

These conditions can lead to serious complications like heart attacks or strokes. Maintaining heart health through proper diet, regular exercise, and routine check-ups is essential.

---

## Types of Cardiovascular Disease
### 1. Arrhythmia
Arrhythmia is an irregular heart rhythm caused by malfunctioning electrical signals in the heart, leading to abnormally fast, slow, or irregular heartbeats.

### 2. Heart Failure
Heart failure occurs when the heart cannot pump blood efficiently, often resulting from coronary heart disease or hypertension.

### 3. Coronary Heart Disease
This happens when coronary arteries are narrowed or blocked by fatty plaque, increasing the risk of a heart attack.

### 4. Congenital Heart Defects
These are structural or functional heart issues present at birth, affecting blood flow.

### 5. Heart Valve Disease
Involves improper opening or closing of heart valves, disrupting normal blood flow.

---

## Features
### 1. Chatbot "Kard AI"
- An intelligent conversational assistant trained to answer questions about cardiovascular health.

### 2. Early Risk Prediction
- Predict the likelihood of cardiovascular disease based on lifestyle factors.

---

## Technologies Used
### Frameworks and Libraries:
- **Flask**: For the web application backend.
- **Flask-CORS**: For handling cross-origin requests.
- **MySQL**: For database management.
- **Torch**: For machine learning.
- **Transformers**: For the text generation model.
- **PEFT**: For model parameter-efficient fine-tuning.
- **Scikit-learn**: For building the random forest prediction model.
- **Joblib**: For model serialization.
- **Pandas**: For data handling.
- **Gdown**: For downloading necessary files.

---

## Models
1. **Risk Prediction**
   - Built using a **Random Forest** model in Scikit-learn.
   - Predicts cardiovascular risk based on user-provided data.

2. **Chatbot "Kard AI"**
   - Fine-tuned **FLAN-T5** model using the `lavita/medical-qa-datasets`.
   - Enables context-aware responses to cardiovascular-related questions.

---

## Dataset
### lavita/medical-qa-datasets
- A comprehensive dataset used to fine-tune the FLAN-T5 model for the chatbot.

---

## Future Plans
- Enhance prediction accuracy with additional models.
- Integrate a feature to track users' health over time.
- Expand chatbot capabilities for broader health education.

---

## Contributions
We welcome contributions! Feel free to open issues or submit pull requests.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

