# Fake Job Posting Detection using Random Forest
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle

df = pd.read_csv("D:/springboard intership 6.0/Classes/fake_job_postings.csv")
df = df[['title', 'description', 'requirements', 'fraudulent']].fillna("")
df['text'] = df['title'] + " " + df['description'] + " " + df['requirements']
df = shuffle(df, random_state=42)

X = df['text']
y = df['fraudulent']

vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X_vec = vectorizer.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42, stratify=y)

model = RandomForestClassifier(n_estimators=150, max_depth=30, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Random Forest Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
