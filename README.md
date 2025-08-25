💻 Usage Guide
Upload Resume: Click the "Choose a Resume File" button and upload a .pdf or .txt resume document.
View Content: The resume text content will appear in a text area for review.
Analyze: Click the "Analyze Resume" button. A spinner will appear as the text is vectorized using DistilBERT and passed through the classifier.
Review Results: The application will display:
The Top 3 Suggested Job Categories with a visual confidence bar chart.
A list of key technical skills detected within the resume text.
🗓 Project Development Timeline Summary
This project was structured across a four-week development cycle:
Week	Focus Area	Technologies
Week 1	Data Preparation & Baseline Features	NLTK, Spacy, TF-IDF
Week 2	Baseline Modeling & Evaluation	Scikit-learn (Naive Bayes, SVM, Logistic Regression)
Week 3	Model Upgrade	DistilBERT (Contextual Embeddings, GPU optimization)
Week 4	Final Deployment & Presentation	Streamlit, Joblib (Model Persistence)
🧠 Model Details
The final classification system relies on a two-step process:
Feature Generation (DistilBERT): The resume text is tokenized and passed through a pre-trained DistilBERT model. The output tokens are averaged to create a single, high-density vector (embedding) that captures the semantic meaning and context of the entire resume.
Classification (Logistic Regression): A standard Logistic Regression model is trained on these high-quality embeddings to predict the job category. This combination provides excellent accuracy and generalization capability.
🔮 Future Enhancements
Fine-tuning BERT: Instead of just using BERT embeddings as features, fine-tune the BERT model itself for the classification task for potentially higher accuracy.
Custom Skill Extraction: Integrate a more advanced Named Entity Recognition (NER) model (e.g., via Spacy) for reliable and customizable skill extraction, rather than relying on a static keyword list.
User Feedback Loop: Implement a simple database to store user feedback on predictions, allowing the model to be retrained periodically with new, validated data.
Deployment Scaling: Migrate the Streamlit app to a cloud platform (e.g., AWS EC2 or Streamlit Cloud) for public access and scalable performance.
