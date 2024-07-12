# app.py

from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

app = Flask(__name__)

# Load the CSV data
dataframe = pd.read_csv('updated1.csv')

# Preprocessing the dataframe
# Assuming you have categorical columns like 'quota', 'pool', 'category' that need encoding
# For simplicity, I'll encode these columns using categorical codes
dataframe['quota'] = pd.Categorical(dataframe['quota']).codes
dataframe['pool'] = pd.Categorical(dataframe['pool']).codes
dataframe['category'] = pd.Categorical(dataframe['category']).codes

# Mapping encoded college codes to names
college_names = dict(zip(dataframe['College'], dataframe['institute_short'] + '-' + dataframe['program_name'] + '-' + dataframe['program_duration']))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        user_rank = int(request.form['rank'])
        user_category = int(request.form['category'])
        user_round = int(request.form['round'])
        user_quota = int(request.form['quota'])
        user_pool = int(request.form['pool'])

        # Prepare the data for prediction
        X = dataframe[['round_no', 'quota', 'pool', 'category', 'opening_rank', 'closing_rank']].values
        y = dataframe['College'].values

        # Decision Tree Classifier function
        def decision_tree(X, y, user_round, user_quota, user_pool, user_category, user_rank):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            clf = DecisionTreeClassifier()
            clf.fit(X_train, y_train)
            return clf.predict([[user_round, user_quota, user_pool, user_category, user_rank, 0]])[0]

        # Predict college
        predicted_college_code = decision_tree(X, y, user_round, user_quota, user_pool, user_category, user_rank)

        # Retrieve college name using the mapping dictionary
        predicted_college_name = college_names.get(predicted_college_code)

        # KMeans clustering for similar colleges
        X_cluster = dataframe[['round_no', 'quota', 'pool', 'category', 'closing_rank']].values
        kmeans = KMeans(n_clusters=5, random_state=42).fit(X_cluster)

        cluster_label = kmeans.predict([[user_round, user_quota, user_pool, user_category, user_rank]])[0]

        cluster_colleges = dataframe[kmeans.labels_ == cluster_label]
        cluster_colleges = cluster_colleges.sort_values('closing_rank').head(10)['College'].map(college_names)

        return render_template('html_output.html', college=predicted_college_name, cluster_colleges=cluster_colleges)

    return render_template('form.html')

if __name__ == '__main__':
    app.run(debug=True)
