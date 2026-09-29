# 🎫 AI Customer Support Ticket Triage

<p align="center">
  <strong>An End-to-End Machine Learning System for Intelligent Customer Support Ticket Triage</strong>
</p>

<p align="center">
  Classify • Prioritize • Route • Review
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python">
  <img src="https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikit-learn" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/NLP-TF--IDF-green" alt="NLP">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit" alt="Streamlit">
  <img src="https://img.shields.io/badge/Status-Completed-success" alt="Status">
</p>

---

## 📌 Project Overview

Customer support teams receive thousands of tickets that need to be **classified, prioritized, and assigned** to the correct support team.

This project develops an **AI-powered customer support ticket triage system** that automates the initial ticket-handling process using **Natural Language Processing and Machine Learning**.

The system analyzes the text of an incoming customer-support ticket and:

- 🏷️ Predicts the ticket category
- 🚨 Predicts the urgency level
- 📊 Calculates model confidence
- 🧭 Determines the appropriate support team
- 👨‍💻 Sends uncertain predictions to human review
- 📝 Maintains a human-review log
- 🖥️ Provides an interactive Streamlit dashboard

The system follows a **human-in-the-loop architecture**, meaning uncertain predictions are not automatically routed.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🏷️ Category Classification | Classifies tickets into five categories |
| 🚨 Urgency Prediction | Predicts Low, Medium, or High urgency |
| 📊 Confidence Estimation | Calculates confidence for both predictions |
| 🧭 Automatic Routing | Routes sufficiently confident tickets |
| 👨‍💻 Human Review | Sends low-confidence tickets for manual review |
| 📝 Review Logging | Stores review cases in a CSV file |
| 🖥️ Streamlit Dashboard | Interactive interface for ticket analysis |
| 📈 Model Evaluation | Accuracy, precision, recall, F1-score and confusion matrices |
| 🔬 Model Comparison | Compares Logistic Regression with Linear SVM |

---

# 🧠 How the System Works

```text
                    CUSTOMER SUPPORT TICKET
                              │
                              ▼
                    ┌───────────────────┐
                    │  Text Processing  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   TF-IDF Vector   │
                    │   Representation  │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
        ┌─────────────────┐       ┌─────────────────┐
        │ Category Model  │       │ Urgency Model   │
        │   Logistic      │       │   Logistic      │
        │   Regression    │       │   Regression    │
        └────────┬────────┘       └────────┬────────┘
                 │                         │
                 ▼                         ▼
        Category + Confidence      Urgency + Confidence
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Confidence Check  │
                    │      ≥ 60% ?      │
                    └─────────┬─────────┘
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
          ┌──────────────┐          ┌──────────────┐
          │   Automatic  │          │    Human     │
          │    Routing   │          │    Review    │
          └──────┬───────┘          └──────────────┘
                 │
                 ▼
          SUPPORT TEAM
          