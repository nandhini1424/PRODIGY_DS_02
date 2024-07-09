import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
titanic_data = sns.load_dataset('titanic')

# Handling missing values in Age and Embarked columns
titanic_data['age'].fillna(titanic_data['age'].median(), inplace=True)
titanic_data['embarked'].fillna(titanic_data['embarked'].mode()[0], inplace=True)

# Drop unnecessary columns like PassengerId, Name, Ticket, and Cabin
titanic_data.drop(['who', 'deck', 'alive', 'alone'], axis=1, inplace=True)

# Survival count plot
plt.figure(figsize=(6, 4))
sns.countplot(x='survived', data=titanic_data)
plt.title('Count of Passengers by Survival')
plt.xlabel('Survived')
plt.ylabel('Count')
plt.show()
print("Survival Count plot generated.")

# Survival rate by sex
plt.figure(figsize=(6, 4))
sns.barplot(x='sex', y='survived', data=titanic_data)
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.show()
print("Survival Rate by Sex plot generated.")

# Survival rate by passenger class
plt.figure(figsize=(6, 4))
sns.barplot(x='class', y='survived', data=titanic_data)
plt.title('Survival Rate by Passenger Class')
plt.xlabel('Passenger Class')
plt.ylabel('Survival Rate')
plt.show()
print("Survival Rate by Passenger Class plot generated.")

# Age distribution of passengers
plt.figure(figsize=(8, 6))
sns.histplot(titanic_data['age'], bins=20, kde=True)
plt.title('Age Distribution of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()
print("Age Distribution of Passengers plot generated.")

# Survival rate by age group
titanic_data['age_group'] = pd.cut(titanic_data['age'], bins=[0, 18, 35, 50, 80], labels=['0-18', '18-35', '35-50', '50+'])
plt.figure(figsize=(8, 6))
sns.barplot(x='age_group', y='survived', data=titanic_data)
plt.title('Survival Rate by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Survival Rate')
plt.show()
print("Survival Rate by Age Group plot generated.")

# Correlation matrix
plt.figure(figsize=(8, 6))
corr_matrix = titanic_data.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('Correlation Matrix')
plt.show()
print("Correlation Matrix plot generated.")
