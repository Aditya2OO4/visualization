# Install Streamlit: pip install streamlit pandas plotly seaborn scikit-learn
import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import requests
import json

# File upload
st.title("Advanced Data Visualizer")
st.sidebar.title("Menu")

# Sidebar menu for additional features
menu = st.sidebar.radio(
    "Navigate",
    [None, "Machine Learning", "Data Preprocessing"],
    index=0,
    format_func=lambda x: "Select an Option" if x is None else x,
)

# Upload Dataset (Always Visible)
st.header("Upload Your Dataset")
uploaded_file = st.file_uploader("Upload your dataset (CSV/Excel)", type=["csv", "xlsx"])

if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Display the dataset
    st.write("### Data Preview")
    st.dataframe(df)

    # Dataset Summary
    with st.expander("Dataset Summary"):
        st.write("### Dataset Summary")
        st.write(f"**Number of Rows:** {df.shape[0]}")
        st.write(f"**Number of Columns:** {df.shape[1]}")
        st.write("**Column Data Types:**")
        st.write(df.dtypes)

    # Visualization (Always Visible)
    st.header("Visualization")

    # Ensure the dataset is not empty
    if df.empty:
        st.error("The dataset is empty. Please upload a valid dataset.")
    else:
        # Select Chart Type
        chart_type = st.selectbox(
            "Select Chart Type",
            [
                "Scatter Plot", "Line Chart", "Bar Chart", "Histogram", "Box Plot", "Pie Chart",
                "Area Chart", "Heatmap", "Pair Plot", "Correlation Heatmap", "Violin Plot", "Sunburst", "Treemap"
            ]
        )

        # Help Section for Each Chart Type
        with st.expander("Help: What does this chart do?"):
            if chart_type == "Scatter Plot":
                st.write("A scatter plot displays points for two variables to show their relationship.")
            elif chart_type == "Line Chart":
                st.write("A line chart connects data points with a line, often used for time series data.")
            elif chart_type == "Bar Chart":
                st.write("A bar chart represents categorical data with rectangular bars.")
            elif chart_type == "Histogram":
                st.write("A histogram shows the distribution of a numeric variable.")
            elif chart_type == "Box Plot":
                st.write("A box plot shows the distribution of a numeric variable and highlights outliers.")
            elif chart_type == "Pie Chart":
                st.write("A pie chart represents proportions of categories as slices of a circle.")
            elif chart_type == "Area Chart":
                st.write("An area chart is similar to a line chart but with the area below the line filled.")
            elif chart_type == "Heatmap":
                st.write("A heatmap shows correlations or relationships between variables using colors.")
            elif chart_type == "Pair Plot":
                st.write("A pair plot shows scatter plots for all pairs of numeric variables.")
            elif chart_type == "Correlation Heatmap":
                st.write("A correlation heatmap shows the correlation coefficients between numeric variables.")
            elif chart_type == "Violin Plot":
                st.write("A violin plot shows the distribution of a numeric variable and its density.")
            elif chart_type == "Sunburst":
                st.write("A sunburst chart shows hierarchical data as concentric circles.")
            elif chart_type == "Treemap":
                st.write("A treemap shows hierarchical data as nested rectangles.")

        # Select X-axis Column
        x_col = st.selectbox("Select X-axis Column", df.columns)

        # Select Y-axis Column (only for applicable chart types)
        y_col = st.selectbox(
            "Select Y-axis Column",
            df.columns if chart_type not in ["Histogram", "Pie Chart", "Heatmap", "Pair Plot", "Correlation Heatmap", "Sunburst", "Treemap"] else [None]
        ) if chart_type not in ["Histogram", "Pie Chart", "Heatmap", "Pair Plot", "Correlation Heatmap", "Sunburst", "Treemap"] else None

        # Select Color Option (Optional)
        if chart_type not in ["Heatmap", "Pair Plot", "Correlation Heatmap"]:
            color_option = st.selectbox("Select Color Column", [None] + list(df.columns))
           
        # Generate Chart
        if st.button("Generate Chart"):
            try:
                # Scatter Plot
                if chart_type == "Scatter Plot":
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_option)

                # Line Chart
                elif chart_type == "Line Chart":
                    fig = px.line(df, x=x_col, y=y_col, color=color_option)

                # Bar Chart
                elif chart_type == "Bar Chart":
                    fig = px.bar(df, x=x_col, y=y_col, color=color_option)

                # Histogram
                elif chart_type == "Histogram":
                    fig = px.histogram(df, x=x_col, color=color_option)

                # Box Plot
                elif chart_type == "Box Plot":
                    fig = px.box(df, x=x_col, y=y_col, color=color_option)

                # Pie Chart
                elif chart_type == "Pie Chart":
                    fig = px.pie(df, names=x_col, color=color_option)

                # Area Chart
                elif chart_type == "Area Chart":
                    fig = px.area(df, x=x_col, y=y_col, color=color_option)

                # Heatmap
                elif chart_type == "Heatmap":
                    numeric_df = df.select_dtypes(include=["number"])
                    if numeric_df.empty:
                        st.error("No numeric columns available for the heatmap.")
                    else:
                        corr = numeric_df.corr()
                        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust dimensions
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)

                # Pair Plot
                elif chart_type == "Pair Plot":
                    numeric_cols = df.select_dtypes(include=["number"]).columns
                    if numeric_cols.empty:
                        st.error("No numeric columns available for the pair plot.")
                    else:
                        fig = px.scatter_matrix(df, dimensions=numeric_cols)
                        st.plotly_chart(fig)

                # Correlation Heatmap
                elif chart_type == "Correlation Heatmap":
                    numeric_df = df.select_dtypes(include=["number"])
                    if numeric_df.empty:
                        st.error("No numeric columns available for the correlation heatmap.")
                    else:
                        corr = numeric_df.corr()
                        fig, ax = plt.subplots(figsize=(10, 8))  # Adjust dimensions
                        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)

                # Violin Plot
                elif chart_type == "Violin Plot":
                    fig = px.violin(df, x=x_col, y=y_col, color=color_option, box=True, points="all")

                # Sunburst
                elif chart_type == "Sunburst":
                    fig = px.sunburst(df, path=[x_col, y_col])

                # Treemap
                elif chart_type == "Treemap":
                    fig = px.treemap(df, path=[x_col, y_col])

                # Display the Plotly Chart
                st.plotly_chart(fig)

            except Exception as e:
                st.error(f"An error occurred while generating the chart: {e}")

    # Correlation & Pair Plot
    if menu == "Correlation & Pair Plot":
        st.header("Correlation & Pair Plot")

        # Automatically apply label encoding to categorical columns
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        st.success("Label encoding applied to all categorical columns!")

        # Display label encoding mappings
        with st.expander("View Label Encoding Mappings"):
            for col, le in label_encoders.items():
                st.write(f"**Column:** {col}")
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.write(mapping)

        # Correlation Heatmap
        with st.expander("Correlation Heatmap"):
            st.write("### Correlation Heatmap")
            corr = df.corr()
            fig, ax = plt.subplots()
            sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Pair Plot
        with st.expander("Pair Plot"):
            st.write("### Pair Plot")
            numeric_cols = df.select_dtypes(include=["number"]).columns
            fig = px.scatter_matrix(df, dimensions=numeric_cols)
            st.plotly_chart(fig)

    # Machine Learning
    if menu == "Machine Learning":
        st.header("Run Machine Learning Models")

        # Automatically apply label encoding to categorical columns
        label_encoders = {}
        for col in df.select_dtypes(include=["object"]).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le

        st.success("Label encoding applied to all categorical columns!")

        # Display label encoding mappings
        with st.expander("View Label Encoding Mappings"):
            for col, le in label_encoders.items():
                st.write(f"**Column:** {col}")
                mapping = dict(zip(le.classes_, le.transform(le.classes_)))
                st.write(mapping)

        # Target Column Selection
        target_col = st.selectbox("Select Target Column", df.columns)

        # Automatically Select Feature Columns
        feature_cols = st.multiselect(
            "Select Feature Columns",
            [col for col in df.columns if col != target_col],
            default=[col for col in df.columns if col != target_col]
        )

        # Suggest a Model
        if st.button("Suggest a Model"):
            if df[target_col].nunique() <= 10:  # Categorical target column
                st.info("Suggested Models: Logistic Regression, Random Forest Classifier, SVM")
            else:  # Numerical target column
                st.info("Suggested Models: Linear Regression, Random Forest Regressor, Gradient Boosting")

        # Select Machine Learning Model
        model_type = st.selectbox(
            "Select Model",
            [
                "Linear Regression", "Logistic Regression", "Decision Tree", "Random Forest",
                "SVM", "K-Nearest Neighbors", "Naive Bayes", "Gradient Boosting", "XGBoost", "Custom Model"
            ]
        )

        # Hyperparameter Tuning
        if model_type in ["Decision Tree", "Random Forest", "Gradient Boosting", "XGBoost"]:
            st.subheader("Hyperparameter Tuning")
            if model_type == "Decision Tree":
                max_depth = st.slider("Max Depth", 1, 20, 5)
                model = DecisionTreeRegressor(max_depth=max_depth)
            elif model_type == "Random Forest":
                n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                max_depth = st.slider("Max Depth", 1, 20, 5)
                model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            elif model_type == "Gradient Boosting":
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                model = GradientBoostingRegressor(learning_rate=learning_rate, n_estimators=n_estimators)
            elif model_type == "XGBoost":
                import xgboost as xgb
                learning_rate = st.slider("Learning Rate", 0.01, 0.5, 0.1)
                n_estimators = st.slider("Number of Estimators", 10, 100, 50)
                model = xgb.XGBRegressor(learning_rate=learning_rate, n_estimators=n_estimators)

        # Custom Model
        if model_type == "Custom Model":
            st.subheader("Upload Your Custom Model")
            uploaded_model = st.file_uploader("Upload a pre-trained model (Pickle file)", type=["pkl"])
            if uploaded_model:
                import joblib
                model = joblib.load(uploaded_model)
                st.success("Custom model loaded successfully!")

        # Train Model
        if st.button("Train Model"):
            X = df[feature_cols]
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train the selected model
            if model_type not in ["Custom Model"]:
                if model_type == "Linear Regression":
                    model = LinearRegression()
                elif model_type == "Logistic Regression":
                    from sklearn.linear_model import LogisticRegression
                    model = LogisticRegression()
                elif model_type == "SVM":
                    from sklearn.svm import SVC
                    model = SVC()
                elif model_type == "K-Nearest Neighbors":
                    from sklearn.neighbors import KNeighborsClassifier
                    n_neighbors = st.slider("Number of Neighbors", 1, 20, 5)
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                elif model_type == "Naive Bayes":
                    from sklearn.naive_bayes import GaussianNB
                    model = GaussianNB()

                model.fit(X_train, y_train)

            # Make Predictions
            predictions = model.predict(X_test)

            # Display Model Performance
            with st.expander("View Evaluation Metrics"):
                st.write("### Model Performance")
                if model_type in ["Logistic Regression", "SVM", "K-Nearest Neighbors", "Naive Bayes"]:
                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
                    st.write(f"Accuracy: {accuracy_score(y_test, predictions)}")
                    st.write(f"Precision: {precision_score(y_test, predictions, average='weighted')}")
                    st.write(f"Recall: {recall_score(y_test, predictions, average='weighted')}")
                    st.write(f"F1-Score: {f1_score(y_test, predictions, average='weighted')}")
                    st.write("Confusion Matrix:")
                    st.write(confusion_matrix(y_test, predictions))
                else:
                    st.write(f"Mean Squared Error: {mean_squared_error(y_test, predictions)}")
                    st.write(f"R² Score: {r2_score(y_test, predictions)}")

            # Display Predicted vs Actual Values
            st.write("### Predicted vs Actual Values")
            results = pd.DataFrame({"Actual": y_test, "Predicted": predictions})
            st.dataframe(results)

        # Save Model
        if st.button("Save Model"):
            import joblib
            joblib.dump(model, "trained_model.pkl")
            st.success("Model saved as 'trained_model.pkl'")

    # Data Preprocessing
    if menu == "Data Preprocessing":
        st.header("Data Preprocessing")

        # Data Description
        with st.expander("Data Description"):
            st.write("### Dataset Description")
            st.dataframe(df.describe())

        # Handle Missing Values
        if df.isnull().values.any():
            with st.expander("Handle Missing Values"):
                st.write("### Handle Missing Values")
                missing_option = st.radio(
                    "Choose how to handle missing values:",
                    ["Drop Rows", "Drop Columns", "Fill with Mean", "Fill with Median", "Fill with Mode", "Custom Value"]
                )
                if missing_option == "Custom Value":
                    custom_value = st.text_input("Enter Custom Value")
                if st.button("Apply Missing Value Handling"):
                    if missing_option == "Drop Rows":
                        df = df.dropna()
                    elif missing_option == "Drop Columns":
                        df = df.dropna(axis=1)
                    elif missing_option == "Fill with Mean":
                        df = df.fillna(df.mean())
                    elif missing_option == "Fill with Median":
                        df = df.fillna(df.median())
                    elif missing_option == "Fill with Mode":
                        df = df.fillna(df.mode().iloc[0])
                    elif missing_option == "Custom Value":
                        df = df.fillna(custom_value)
                    st.success("Missing values handled successfully!")
                    st.dataframe(df)
        else:
            st.info("No null values present in the dataset!")

        # Handle Outliers
        with st.expander("Handle Outliers"):
            st.write("### Detect and Remove Outliers")
            outlier_cols = st.multiselect("Select Columns for Outlier Detection", df.select_dtypes(include=["number"]).columns)
            select_all_outliers = st.checkbox("Select All Columns for Outlier Detection")
            if select_all_outliers:
                outlier_cols = df.select_dtypes(include=["number"]).columns.tolist()
            outlier_method = st.radio("Choose Outlier Detection Method", ["IQR", "Z-Score"])
            if st.button("Remove Outliers"):
                if outlier_method == "IQR":
                    for col in outlier_cols:
                        Q1 = df[col].quantile(0.25)
                        Q3 = df[col].quantile(0.75)
                        IQR = Q3 - Q1
                        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]
                elif outlier_method == "Z-Score":
                    from scipy.stats import zscore
                    for col in outlier_cols:
                        df = df[(zscore(df[col]) < 3).all(axis=1)]
                st.success("Outliers removed successfully!")
                st.dataframe(df)
        # Scaling and Normalization
        with st.expander("Scaling and Normalization"):
            st.write("### Scale or Normalize Data")
            scale_cols = st.multiselect("Select Columns to Scale/Normalize", df.select_dtypes(include=["number"]).columns)
            select_all = st.checkbox("Select All Columns for Scaling")
            if select_all:
                scale_cols = df.select_dtypes(include=["number"]).columns.tolist()
            scaling_method = st.radio("Select Scaling Method", ["Standard Scaling", "Min-Max Scaling", "Robust Scaling"])
            if st.button("Apply Scaling"):
                from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
                scaler = StandardScaler() if scaling_method == "Standard Scaling" else MinMaxScaler() if scaling_method == "Min-Max Scaling" else RobustScaler()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
                st.success("Scaling applied successfully!")
                st.dataframe(df)

        # Encoding Categorical Variables
        with st.expander("Encoding Categorical Variables"):
            st.write("### Encode Categorical Variables")
            cat_cols = st.multiselect("Select Categorical Columns", df.select_dtypes(include=["object"]).columns)
            select_all_cat = st.checkbox("Select All Categorical Columns for Encoding")
            if select_all_cat:
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
            encoding_method = st.radio("Choose Encoding Method", ["Label Encoding", "One-Hot Encoding"])
            if st.button("Apply Encoding"):
                if encoding_method == "Label Encoding":
                    from sklearn.preprocessing import LabelEncoder
                    for col in cat_cols:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col])
                elif encoding_method == "One-Hot Encoding":
                    df = pd.get_dummies(df, columns=cat_cols)
                st.success("Encoding applied successfully!")
                st.dataframe(df)

        # Feature Engineering
        with st.expander("Feature Engineering"):
            st.write("### Create New Features")
            new_col_name = st.text_input("Enter New Column Name")
            formula = st.text_input("Enter Formula (e.g., col1 + col2)")
            if st.button("Create New Feature"):
                try:
                    df[new_col_name] = eval(formula)
                    st.success(f"New column '{new_col_name}' created successfully!")
                    st.dataframe(df)
                except Exception as e:
                    st.error(f"Error: {e}")

        # Data Cleaning
        with st.expander("Data Cleaning"):
            st.write("### Data Cleaning Options")
            if st.button("Remove Duplicates"):
                df = df.drop_duplicates()
                st.success("Duplicates removed successfully!")
                st.dataframe(df)
            if st.checkbox("Rename Columns"):
                col_to_rename = st.selectbox("Select Column to Rename", df.columns)
                new_col_name = st.text_input("Enter New Column Name")
                if st.button("Rename Column"):
                    df.rename(columns={col_to_rename: new_col_name}, inplace=True)
                    st.success(f"Column renamed to {new_col_name}!")
                    st.dataframe(df)

    # Chatbot Section
    st.header("AI Chatbot Assistant (Llama 3.2)")

    # User Query Input
    user_query = st.text_input("Ask the AI anything about your dataset or visualizations:")

    # Ollama API Configuration
    OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
    MODEL_NAME = "llama3.2"

    # Generate Dataset Context
    if uploaded_file:
        dataset_context = f"""
        The dataset has {df.shape[0]} rows and {df.shape[1]} columns.
        The first few rows of the dataset are:
        {df.head(5).to_string(index=False)}.
        """
    else:
        dataset_context = "No dataset has been uploaded."

    # Chatbot Response
    if user_query:
        try:
            prompt = f"""
            You are an AI assistant. Here is the context of the dataset:
            {dataset_context}

            User Query: {user_query}
            """

            response = requests.post(
                OLLAMA_API_URL,
                json={"model": "llama3.2", "prompt": prompt},
                timeout=30  # Set timeout
            )

            if response.status_code == 200:
                raw_responses = response.text.splitlines()
                ai_response = ""
                for raw in raw_responses:
                    try:
                        response_json = json.loads(raw)
                        ai_response += response_json.get("response", "")
                    except json.JSONDecodeError:
                        continue

                if ai_response:
                    st.write(f"**AI Response:** {ai_response}")
                else:
                    st.error("The AI did not return any content. Please try again.")
            else:
                st.error(f"Error: Unable to connect to the Ollama API. Status Code: {response.status_code}")
        except requests.exceptions.Timeout:
            st.error("The request timed out. Please try again.")
        except Exception as e:
            st.error(f"An error occurred while processing your query: {e}")
else:
    st.warning("Please upload a dataset to proceed!")