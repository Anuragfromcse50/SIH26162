# SIH26162 - Fire & Landcover Detection Project

## 📌 Project Overview
Ye project SIH (Smart India Hackathon) problem statement SIH26162 ke liye banaya gaya hai. Iska main maksad satellite data aur thermal sources ka use karke fire aur landcover ka pata lagana hai.

## 🛠️ Technologies Used
- Python
- Jupyter Notebook (`test.ipynb`)
- Data Analysis (CSV files)
- Machine Learning / Classification

## 📂 Folder Structure
- `app/` - Isme project ka main Python code (`app.py`) hai.
- `data/` - Isme saari CSV files hain (jaise `classified_fire_data.csv`, `land_cover.csv`, etc.).
- `test.ipynb` - Data analysis aur model testing ki notebook.
- `requirements.txt` - Project chalane ke liye zaroori Python libraries ki list.
- `SIH26162_final_map.html` - Final output ka map.

## 🚀 How to Run
1. Is repository ko clone karein.
2. Terminal mein `pip install -r requirements.txt` chalayein.
3. `test.ipynb` ko Jupyter Notebook mein khol kar run karein.


**“Currently, we have implemented a rule-based classification approach using engineered features such as persistence, distance to industrial sites, and land-cover information. Random Forest ML classification is the next stage, where we will train and evaluate the model using train/test split, accuracy, and confusion matrix.”**

🟢 Data + GIS + Feature Engineering + Rule-Based Classification = Done
🟡 Random Forest + Train/Test + Accuracy + Confusion Matrix = Abhi karna hai
🟡 Dashboard = Uske baad/final integration
