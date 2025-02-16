import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the pre-trained similarity matrix and course data
similarity = pickle.load(open('similarity.pkl', 'rb'))
course_dict = pickle.load(open('course_list.pkl', 'rb'))

# Recreate the DataFrame from the dictionary
new_df = pd.DataFrame.from_dict(course_dict)

# Sidebar design
st.sidebar.image("https://via.placeholder.com/150", caption="Course Recommendation System", use_column_width=True)
st.sidebar.markdown("### Explore courses tailored to your preferences!")

# App title
st.title("🎓 Coursera Course Recommendation System")
st.markdown(
    """
    Welcome to the **Coursera Course Recommendation System**!  
    Find courses similar to your interests and expand your learning journey.
    """
)

# Selectbox for course selection
st.markdown("### Select a Course to Get Started")
course_name = st.selectbox(
    "Choose from the dropdown below:",
    new_df['course_name'].values,
    help="Select a course to see recommendations tailored to it."
)

# Recommendation logic
def recommend(course):
    # Find index of selected course
    course_index = new_df[new_df['course_name'] == course].index[0]
    distances = similarity[course_index]
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]

    # Display recommended courses and URLs
    recommendations = []
    for i in course_list:
        recommended_course = new_df.iloc[i[0]]
        recommendations.append((recommended_course.course_name, recommended_course['Course URL']))
    return recommendations

# Show recommendations
if st.button("🔍 Show Recommended Courses"):
    st.markdown(f"### 🔗 Courses Similar to **{course_name}**:")
    recommendations = recommend(course_name)
    for course, url in recommendations:
        st.markdown(f"🔸 **[{course}]({url})**")
        st.markdown("---")
else:
    st.info("👈 Select a course and click the button to see recommendations!")

# Footer
st.markdown("---")
st.markdown(
    """
    👨‍💻 Devloped By [Chinna Durai](https://github.com/durai1937) |  

    """
)
