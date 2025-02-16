import pandas as pd


data = pd.read_csv(r"Coursera.csv")

# Select the necessary columns, assuming 'Course URL' exists in the dataset
data = data[['Course Name', 'Difficulty Level', 'Course Description', 'Skills', 'Course URL']]

# Clean 'Course Name', 'Course Description', and 'Skills' columns
data['Course Name'] = data['Course Name'].str.replace(' ', ',').str.replace(',,', ',').str.replace(':', '')
data['Course Description'] = data['Course Description'].str.replace(' ', ',').str.replace(',,', ',').str.replace('_', '').str.replace(':', '').str.replace('(', '').str.replace(')', '')
data['Skills'] = data['Skills'].str.replace('(', '').str.replace(')', '')

# Create a 'tags' column
data['tags'] = data['Course Name'] + data['Difficulty Level'] + data['Course Description'] + data['Skills']
new_df = data[['Course Name', 'Course URL', 'tags']]
new_df['tags'] = new_df['tags'].str.replace(',', ' ')
new_df['Course Name'] = new_df['Course Name'].str.replace(',', ' ')
new_df.rename(columns={'Course Name': 'course_name'}, inplace=True)

# Convert 'tags' to lowercase and apply stemming
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    return " ".join([ps.stem(i) for i in text.split()])

new_df['tags'] = new_df['tags'].apply(stem)
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

# Modify the recommend function to include course URLs
def recommend(course):
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    for i in course_list:
        recommended_course = new_df.iloc[i[0]]
        print(f"{recommended_course.course_name} - {recommended_course['Course URL']}")

# Example usage
recommend('Business Strategy Business Model Canvas Analysis with Miro')

# Save the similarity matrix and course data for later use
import pickle
pickle.dump(similarity, open('similarity.pkl', 'wb'))
pickle.dump(new_df.to_dict(), open('course_list.pkl', 'wb'))
pickle.dump(new_df, open('courses.pkl', 'wb'))
