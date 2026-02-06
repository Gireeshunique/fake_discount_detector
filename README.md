### 🛒 Fake Discount Detector (Amazon E-commerce)
#####📌 Project Overview

Online shopping platforms often display large discounts that may not always be genuine.
This project builds a machine learning system that detects whether an e-commerce discount is REAL or FAKE using historical pricing patterns.

The system analyzes:

Original price

Discounted price

Category-wise historical averages

Discount percentage

It then predicts the probability of a fake discount and provides consumer-friendly insights via a Streamlit web application.

##### 🎯 Objectives

Detect misleading e-commerce discounts

Protect consumers from fake pricing strategies

Apply end-to-end machine learning workflow

Deploy a real-time prediction web app

##### 📂 Dataset

Source: Amazon Products Price Dataset from Kaggle

Size: ~10,000–25,000 products

Format: CSV

Key Columns
Column	Description
product_name	Product title
category	Product category
actual_price	Original listed price
discounted_price	Sale price
discount_percentage	Discount offered

Historical price is simulated using category-level average pricing, which is acceptable for beginner ML projects.

##### 🧠 Machine Learning Workflow

Data loading and cleaning

Price normalization (₹ symbol removal)

Feature engineering

Fake discount labeling

Model training (Logistic Regression)

Model evaluation

Model serialization

Streamlit deployment

##### ⚙️ Feature Engineering

Discount Percentage

Price Inflation

Discount vs Historical Price

Category Encoding

Target Variable
fake_discount = 1 → Fake Discount
fake_discount = 0 → Real Discount

##### 🤖 Model Used

Algorithm: Logistic Regression

Reason:

Simple

Explainable

Beginner-friendly

Interview-safe

##### 📊 Model Evaluation

Accuracy score

Confusion matrix

Classification report

##### 🖥️ Web Application (Streamlit)

The Streamlit app allows users to:

Enter product price details

Select product category

Predict fake discount probability

View consumer-friendly insights

##### 📁 Project Structure
fake-discount-detector/
│
├── data/
│   └── amazon.csv
│
├── models/
│   ├── model.pkl
│   └── encoder.pkl
│
├── src/
│   ├── data_preprocessing.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
└── README.md

##### 🚀 How to Run the Project
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Train the Model
python train_model.py

3️⃣ Run Streamlit App
streamlit run app.py

##### 💡 Example Output

Fake Discount Probability: 72%

Verdict: ❌ Fake Discount

Consumer Tips: Compare prices across platforms

##### 🧪 Technologies Used

Python

Pandas & NumPy

Scikit-learn

Matplotlib & Seaborn

Streamlit

##### 🎓 Use Cases

Consumer protection tools

Price comparison platforms

E-commerce analytics

Academic ML projects

##### 🏆 Key Highlights

✔ Real Amazon dataset
✔ End-to-end ML pipeline
✔ Explainable predictions
✔ Deployed web application
✔ Beginner-friendly design

##### 📜 License

This project is for educational purposes only.
Dataset credit belongs to the original Kaggle contributors.

##### 🙌 Author

Your Name
Gireesh Boggala
