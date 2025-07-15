# FAQ-Chatbot
AI-powered FAQ Chatbot that answers user queries by training on 100,000+ question-answer pairs using NLP and Machine Learning. Built with Python and deployed with a simple UI.

An intelligent FAQ chatbot system that can understand user questions and respond with the most relevant answer using Natural Language Processing (NLP) and Machine Learning techniques. Trained on over 100,000 real-world Q&A pairs for high accuracy and flexibility.



##  Features

 Accepts user input and returns the best-matching answer  
 Trained on a large dataset of question-answer pairs (100,000+)  
 Uses NLP techniques (TF-IDF, cosine similarity)  
 Built using Python and scikit-learn  
 Simple user interface (GUI or CLI)  
 Easily extendable with your own Q&A dataset



##  How It Works

1. Preprocesses a CSV dataset of FAQs (questions and answers)
2. Converts questions into vector form using TF-IDF
3. When a user enters a query:
   - It finds the most similar question in the dataset
   - Returns the corresponding answer

# Requirements:
- Python 3.10 or later
- pip (Python package manager)
- Internet connection (for first-time NLTK downloads)

# Installation Instructions:

1. Open your terminal / PowerShell and navigate to the project directory:
   cd path/to/faq_chatbot

2. (Optional but recommended) Create and activate a virtual environment:
   python -m venv venv
   venv\Scripts\activate     [for Windows]
   source venv/bin/activate    [for Mac/Linux]

3. Install all required packages:
   pip install -r requirements.txt

# How to Run the Chatbot:

Use the following command to launch the chatbot in your web browser:

streamlit run chatbot_faq_improved.py

OR (if `streamlit` is not globally available):

 python -m streamlit run chatbot_faq_improved.py


# Access the Chatbot:
Once running, the chatbot will open at:
http://localhost:8501


