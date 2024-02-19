import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from plotly.subplots import make_subplots
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import plotly.graph_objects as go
from xgboost import XGBClassifier
from sklearn.decomposition import PCA
import time


class PassengerSatisfactionAnalysis:
    def __init__(self, train_path, test_path):
        self.dataTrain = pd.read_csv(train_path)
        self.dataTest = pd.read_csv(test_path)

    def fill_na_mean(self):
        mean_arrival_delay = self.dataTrain['Arrival Delay in Minutes'].mean()
        self.dataTrain['Arrival Delay in Minutes'].fillna(mean_arrival_delay, inplace=True)
        mean_arrival_delay2 = self.dataTest['Arrival Delay in Minutes'].mean()
        self.dataTest['Arrival Delay in Minutes'].fillna(mean_arrival_delay2, inplace=True)

    def drop_columns(self,colums):
        self.dataTrain = self.dataTrain.drop(columns=colums, axis=1)
        self.dataTest = self.dataTest.drop(columns=colums, axis=1)

    def data_info(self):
        print('-----------Boyutu---------')
        print(self.dataTrain.shape)
        print('-----Sutün--------------Veri Tipi-------- ')
        print(self.dataTrain.dtypes)
        print('--------Benzersiz Değer--------')
        print(self.dataTrain.nunique())
        print('--------Boş Değer----------')
        print(self.dataTrain.isnull().sum())
        print('--------Veri Özeti-------')
        print(self.dataTrain.describe(include='all'))

    def label_encode_categorical_columns(self):
        object_columns = self.dataTrain.select_dtypes(include=['object']).columns
        label_encoder = LabelEncoder()
        for column in object_columns:
            self.dataTrain[column] = label_encoder.fit_transform(self.dataTrain[column])
            self.dataTest[column] = label_encoder.transform(self.dataTest[column])

    def plot_correlation_matrix(self):
        corr_matrix = self.dataTrain.corr()
        plt.figure(figsize=(16, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, cbar_kws={"shrink": 0.9})
        plt.title("Korelasyon Matrisi")
        plt.show()

    def find_duplicate_rows(self):
        duplicated_rows = self.dataTrain[self.dataTrain.duplicated()]
        if not duplicated_rows.empty:
            print("Tekrar eden satırlar:")
            print(duplicated_rows)
        else:
            print("Tekrar eden satırlar bulunmamaktadır.")

    def plot_violin(self, x_column, y_column, category_column, title='', x_title='', y_title=''):
        categories = self.dataTrain[category_column].unique()
        fig = go.Figure()
        for category in categories:
            category_data = self.dataTrain[self.dataTrain[category_column] == category]
            fig.add_trace(go.Violin(x=category_data[x_column],
                                    line_color='lightseagreen' if category == 'satisfied' else 'red',
                                    y0=0, name=f'{category} passengers {y_column}'))
        fig.update_traces(orientation='h', side='positive', meanline_visible=True)
        fig.update_layout(title=f'<b>{title}<b>',
                          titlefont={'size': 20},
                          xaxis_zeroline=False,
                          paper_bgcolor='lightgrey',
                          plot_bgcolor='lightgrey')
        fig.update_xaxes(showgrid=False, title=f'<b>{x_title}<b>')
        fig.update_yaxes(title=f'<b>{y_title}<b>')
        fig.update_traces(opacity=0.7)
        fig.show()

    def plot_histograms_subplots(self, features, nbins=20, title='', title_size=30):
        total_plots = len(features)
        cols = 2
        rows = (total_plots + 1) // cols
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=features)
        for i, feature in enumerate(features, start=1):
            fig.add_trace(go.Histogram(x=self.dataTrain[feature], nbinsx=nbins, name=feature),
                          row=i // cols + 1, col=i % cols + 1)
        fig.update_layout(title_text=f'<b>{title}<b>',
                          title_font=dict(size=title_size),
                          paper_bgcolor='lightgrey',
                          plot_bgcolor='lightgrey',
                          showlegend=False,
                          height=rows * 300)
        fig.show()

    def plt_histogram(self):
        sns.countplot(x='Class', hue='satisfaction', palette="YlOrBr", data=self.dataTrain)
        plt.show()
        with sns.axes_style(style='ticks'):
            g = sns.catplot(x="satisfaction", col="Gender", col_wrap=2, data=self.dataTrain, kind="count", height=4.5,
                            aspect=1.0)
        g.set_axis_labels("Satisfaction", "Count")
        g.set_titles(col_template="{col_name}")
        plt.show()
        with sns.axes_style('white'):
            g = sns.catplot(x="Age", data=self.dataTrain, aspect=3.0, kind='count', hue='satisfaction', order=range(5, 80))
            g.set_ylabels('Count')
            g.set_xlabels('Age vs Passenger Satisfaction')
            plt.show()
        self.dataTrain[['Gender', 'satisfaction']].groupby(['Gender'], as_index=False).mean().sort_values(
            by='satisfaction', ascending=False)

    def prepare_data_for_models(self, apply_pca=False, n_components=None):
        xTrain = self.dataTrain.drop(columns='satisfaction', axis=1)
        xTest = self.dataTest.drop(columns='satisfaction', axis=1)
        yTrain = self.dataTrain['satisfaction']
        yTest = self.dataTest['satisfaction']

        scaler = StandardScaler()
        xTrain = scaler.fit_transform(xTrain)
        xTest = scaler.transform(xTest)

        if apply_pca:
            pca = PCA(n_components=n_components)
            xTrain = pca.fit_transform(xTrain)
            xTest = pca.transform(xTest)

        return xTrain, xTest, yTrain, yTest

    def plot_confusion_matrix(self, cm, class_names):
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.show()

    def evaluate_model(self, model, xTrain, yTrain, xTest, yTest, class_names):
        model.fit(xTrain, yTrain)
        y_pred = model.predict(xTest)
        cm = confusion_matrix(yTest, y_pred)
        self.plot_confusion_matrix(cm, class_names)

        # Accuracy
        acc = model.score(xTest, yTest)
        print(f"Model Accuracy: {acc}")

        # Precision, Recall, F1-Score
        report = classification_report(yTest, y_pred, target_names=class_names)
        print("Classification Report:\n", report)

        precision = precision_score(yTest, y_pred, average='weighted')
        recall = recall_score(yTest, y_pred, average='weighted')
        f1 = f1_score(yTest, y_pred, average='weighted')

        print(f'Weighted Precision: {precision}')
        print(f'Weighted Recall: {recall}')
        print(f'Weighted F1-Score: {f1}')

    def _apply_model_with_timing(self, model, xTrain, yTrain, xTest, yTest, class_names=None):
        start_time = time.time()  # Başlangıç zamanını kaydet
        self.evaluate_model(model, xTrain, yTrain, xTest, yTest, class_names=class_names)
        end_time = time.time()  # Bitiş zamanını kaydet
        elapsed_time = end_time - start_time  # Geçen süreyi hesapla
        print(f"Modelin çalışma süresi: {elapsed_time:.2f} saniye")

    def apply_random_forest(self, xTrain, yTrain, xTest, yTest):
        model = RandomForestClassifier()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_logistic_regression(self, xTrain, yTrain, xTest, yTest):
        model = LogisticRegression()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_support_vector_classifier(self, xTrain, yTrain, xTest, yTest, kernel='linear'):
        model = SVC(kernel=kernel)
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_decision_tree(self, xTrain, yTrain, xTest, yTest):
        model = DecisionTreeClassifier()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_knn(self, xTrain, yTrain, xTest, yTest, n_neighbors=5):
        model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_gaussianNB(self, xTrain, yTrain, xTest, yTest):
        model = GaussianNB()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_gradiendtBoostingClassifier(self, xTrain, yTrain, xTest, yTest):
        model = GradientBoostingClassifier()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_neural_network(self, xTrain, yTrain, xTest, yTest, hidden_layer_sizes=(50, 30), max_iter=500,
                             random_state=42):
        model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=random_state)
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])

    def apply_xgb(self, xTrain, yTrain, xTest, yTest):
        model = XGBClassifier()
        self._apply_model_with_timing(model, xTrain, yTrain, xTest, yTest,
                                      class_names=['neutral or dissatisfied', 'satisfied'])




# Kullanım örneği
train_path = 'C:/Users/ABREBO/Desktop/csv/train.csv/train.csv'
test_path = 'C:/Users/ABREBO/Desktop/csv/test.csv/test.csv'

analysis = PassengerSatisfactionAnalysis(train_path, test_path)
analysis.fill_na_mean()
analysis.drop_columns(['Unnamed: 0', 'id', 'Departure Delay in Minutes', 'Cleanliness'])
#analysis.data_info()
analysis.label_encode_categorical_columns()
#analysis.plot_correlation_matrix()
#nalysis.find_duplicate_rows()
#analysis.plot_violin('Age', 'Seat comfort', 'satisfaction', 'Distribution of satisfaction over Age','Age', 'Seat comfort')
#analysis.plot_histograms_subplots(['Age', 'Class', 'Seat comfort'], title='Distribution of Features',title_size=25)
#analysis.plt_histogram()   

xTrain, xTest, yTrain, yTest = analysis.prepare_data_for_models()
#analysis.apply_random_forest(xTrain, yTrain, xTest, yTest)
#analysis.apply_logistic_regression(xTrain, yTrain, xTest, yTest)

#analysis.apply_support_vector_classifier(xTrain, yTrain, xTest, yTest,kernel='linear')

#analysis.apply_decision_tree(xTrain, yTrain, xTest, yTest)
#analysis.apply_knn(xTrain, yTrain, xTest, yTest,n_neighbors=7)
#analysis.apply_gaussianNB(xTrain, yTrain, xTest, yTest)
#analysis.apply_gradiendtBoostingClassifier(xTrain, yTrain, xTest, yTest)
#analysis.apply_neural_network(xTrain, yTrain, xTest, yTest)
#analysis.apply_xgb(xTrain, yTrain, xTest, yTest)

xTrain, xTest, yTrain, yTest = analysis.prepare_data_for_models(apply_pca=True, n_components=5)
analysis.apply_support_vector_classifier(xTrain, yTrain, xTest, yTest, kernel='linear')