import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def analyze_and_predict(df):
    data = df[['PayloadMass', 'Orbit', 'LaunchSite', 'Flights', 'GridFins', 'Reused', 'Legs', 'LandingPad', 'Block', 'ReusedCount', 'Serial', 'Class']]
    mean = data['PayloadMass'].mean()
    data['PayloadMass'].fillna(mean, inplace=True)
    data['LandingPad'].fillna('Unknown', inplace=True)
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    columns_to_encode = ['Orbit', 'LaunchSite', 'LandingPad', 'Serial']
    ct = ColumnTransformer(
        transformers=[
            ('encoder', OneHotEncoder(), columns_to_encode)
        ],
        remainder='passthrough'
    )

    X = ct.fit_transform(data)
    feature_names = ct.get_feature_names_out()
    data_transformed = pd.DataFrame(X.toarray(), columns=feature_names) # Convert sparse matrix to dense array
    X = data_transformed.iloc[:, :-1].values
    y = data_transformed.iloc[:, -1].values

    result = {
        'Model': [],
        'Accuracy': []
    }

    """### Spliting data"""

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    """### Logistic Regression"""

    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('Logistic Regression')
    result['Accuracy'].append(score)

    """### SVC"""

    model = SVC()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('SVC')
    result['Accuracy'].append(score)

    """### Decision Tree Classifier"""

    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('Decision Tree Classifier')
    result['Accuracy'].append(score)

    """### Random Forest"""

    model = RandomForestClassifier(n_estimators=10)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('Random Forest')
    result['Accuracy'].append(score)

    """### K Neighbours"""

    model = KNeighborsClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('K Neighbours')
    result['Accuracy'].append(score)

    """### GaussianNB"""

    model = GaussianNB()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('GaussianNB')
    result['Accuracy'].append(score)

    """### XGB Classifier"""

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    result['Model'].append('XGB Classifier')
    result['Accuracy'].append(score)

    """### Artificial Neural Network"""

    ann = tf.keras.models.Sequential()
    ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=10, activation='relu'))
    ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    ann.fit(x_train, y_train, batch_size=32, epochs=100)
    ann.evaluate(x_test, y_test)
    ann_pred = ann.predict(x_test)
    ann_pred = (ann_pred > 0.5)
    score = accuracy_score(y_test, ann_pred)
    matrix = confusion_matrix(y_test, ann_pred)
    result['Model'].append('ANN')
    result['Accuracy'].append(score)

    """## Comparing Result"""

    result = pd.DataFrame(result)
    result.to_csv('result.csv', index=True)

    plt.figure(figsize=(12, 6))
    plt.plot(result['Model'], result['Accuracy'], marker='o', linestyle='-', color='#004488')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy by Different Models')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('accuracy_by_model.png')
    plt.show()

    """## Best Model"""

    model = XGBClassifier()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    score = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matrix)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    plt.show()

    # Assuming 'model' is your trained XGBoost model and 'x_test', 'y_test' are your test data
    y_pred_proba = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkblue', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    """## Data Visualization"""
    launchsite_counts = data['Orbit'].value_counts()
    total_launches = launchsite_counts.sum()
    orbit_counts = data['Orbit'].value_counts()
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Blues(np.linspace(1, 0.4, len(orbit_counts)))
    plt.pie(orbit_counts, labels=None, colors=colors, startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.text(0.97, 0.01, f'Total no of launches: {total_launches}', ha='right', va='bottom', transform=plt.gca().transAxes)
    legend_labels = [f'{site}: {count}' for site, count in launchsite_counts.items()]
    plt.legend(legend_labels, title='Orbit', loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title('Distribution of Launches through Orbit')
    plt.axis('equal')
    plt.savefig('orbit_distribution.png')
    plt.show()

    # Filter the DataFrame for the desired orbits
    filtered_data = data[data['Orbit'].isin(['GTO', 'ISS', 'VLEO', 'PO', 'LEO', 'SSO', 'MEO'])]
    # Group the filtered data by Orbit and Class, and count the occurrences
    grouped_data = filtered_data.groupby(['Orbit', 'Class']).size().unstack(fill_value=0)
    # Plot the bar graph
    plt.figure(figsize=(8, 12))
    grouped_data.plot(kind='bar', color=['#004488', '#4477AA'])  # Dark blue for successful, normal blue for not successful
    plt.xlabel('Orbit')
    plt.ylabel('Number of Launches')
    plt.title('Successful and Unsuccessful Launches through Orbit')
    plt.legend(title='Class', labels=['Not Successful', 'Successful'])
    plt.xticks(rotation=0)
    plt.savefig('orbit_launch_success.png')
    plt.show()

    # Group the data by 'LaunchSite' and count the number of launches for each site
    launchsite_counts = data['LaunchSite'].value_counts()
    # Calculate the total number of launches
    total_launches = launchsite_counts.sum()
    # Define colors for the pie chart
    colors = ['#004488', '#2C6699', '#4477AA']
    # Plot the pie chart
    plt.figure(figsize=(8, 6))
    plt.pie(launchsite_counts, labels=launchsite_counts.index, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    # Set title
    plt.title('Number of Launches by Launch Site')
    # Add the total number of launches to the lower right corner
    plt.text(0.95, 0.05, f'Total no of launches: {total_launches}', ha='right', va='bottom', transform=plt.gca().transAxes)
    # Ensure the circle's proportion
    plt.axis('equal')
    plt.savefig('launch_site_distribution.png')
    plt.show()

    # Filter the DataFrame for the desired LaunchSites
    filtered_data = data
    # Group the filtered data by LaunchSite and Class, and count the occurrences
    grouped_data = filtered_data.groupby(['LaunchSite', 'Class']).size().unstack(fill_value=0)
    # Plot the bar graph
    plt.figure(figsize=(10, 8))  # Adjust figure size as needed
    grouped_data.plot(kind='bar', color=['#004488', '#4477AA'])  # Dark blue for successful, normal blue for not successful
    plt.xlabel('Launch Site')
    plt.ylabel('Number of Launches')
    plt.title('Successful and Unsuccessful Launches by Launch Site')
    plt.legend(title='Class', labels=['Not Successful', 'Successful'])
    plt.xticks(rotation=0)
    plt.savefig('launch_site_launch_success.png')
    plt.show()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='Orbit', y='PayloadMass')
    plt.title('Distribution of PayloadMass by Orbit')
    plt.xlabel('Orbit')
    plt.ylabel('PayloadMass')
    plt.savefig('payload_mass_by_orbit.png')
    plt.show()

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=df, x='PayloadMass', y='Flights', hue='Orbit')
    plt.title('Relationship between PayloadMass, Flights, and Orbit')
    plt.xlabel('PayloadMass')
    plt.ylabel('Flights')
    plt.savefig('payload_mass_flights_orbit.png')
    plt.show()

    # Heatmap: Correlation between numerical variables
    numerical_vars = ['PayloadMass', 'Flights', 'ReusedCount', 'Block']
    corr_matrix = df[numerical_vars].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='Blues')
    plt.title('Correlation Matrix')
    plt.savefig('correlation_matrix.png')
    plt.show()