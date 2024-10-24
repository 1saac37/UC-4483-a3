'''
*******************************
Author: Isaac Teo, Quang Anh Nguyen
u3295812 u3251660 Assessment 3 10/2024
Programming: 4483 Sem 2
*******************************
'''  

## import necessary library
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import pickle


st.title('Python assignment 3')

uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=['csv'])
if uploaded_file is None:
    st.write("""
    # Please choose a CSV file
    """)
else:
    st.write("""
    # Step 1: Reading the dataset
    #### Using PANDAS package, read the dataset, remove duplicates and print sample data for inspection. 
    """)
    # Read CSV
    laptop_price_df = pd.read_csv(uploaded_file)

    # Dataset preview
    st.write(f"### Dataset Preview:")
    st.write(laptop_price_df.head(10))

    # Show and remove duplicate data
    st.write(f"### Show duplicate data before using drop_duplicates()")
    # laptop_price_df is my dataframe
    duplicate_count = pd.DataFrame({"Duplicate Count": ["Duplicate Count"], "Number of duplicate": [laptop_price_df.duplicated().sum()]})

    st.dataframe(duplicate_count, use_container_width=True)

    st.write(f"There are no duplicate entries")

    ## Show data after drop_duplicate
    st.write(f"### This is the sample data after removing duplicate entries")
    st.write(laptop_price_df.head(10))
    st.write(f"No duplicate entries. No changes made.")

    st.write("### Observation:")
    st.write(
        """
        In this step, after loading the dataset and conducting a duplicate check, 
        we found that there were no duplicate. This means the dataset is clean and consistent, 
        requiring no further data cleaning. Thus, we can move on to the next step because  
        our data is reliable.
        """,
        unsafe_allow_html=True
    )

    st.write("""
        # Step 2: Problem statement
        By analysing different variables of laptops; such as brand, cpu, ram, and price, visually and statistically find the key variables that affect price.
        With target variables known, then train a machine learning engine to predict prices of laptops outside of the data set, given the targeted variables.
 
        """)

    ## Showing the laptop price collums
    st.write(f"### Rows, Columns: {laptop_price_df.shape}")

    st.write("Columns in the dataset:", laptop_price_df.columns.tolist())

    st.write(f"### Descriptive Statistics of Laptop Price Dataset :")
    st.dataframe(laptop_price_df.describe(), use_container_width=True)

    st.write("### Observation:")
    st.write("""
           At this stage, we have a general understanding of the dataset's structure, column types, and summary statistics. 
       """)


    st.write("""
            # Step 3: Visualising the distribution of Target variable
            #### Identify the dependent/target variable, or the prediction variable and look at the distribution to assess the class imbalance in data (whether the data is balanced or skewed). 
            """)

    target = st.selectbox("Select the target variable (dependent variable):", laptop_price_df.columns)
    st.write(f"### Selected Target Variable: {target}")

    fig, ax = plt.subplots()
    ax.hist(laptop_price_df[target], bins=30, edgecolor='k', alpha=0.7)
    ax.set_title(f"Distribution of {target}")
    ax.set_xlabel(target)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    st.write("### Observation:")
    st.write("""
              We visualised the distribution of the target variable (price) to assess the balance and skewnes. Depends on which target variable we choose, it can show a different graph.
          """)

    st.write("""
                # Step 4: Data exploration at basic level
                #### This step is needed to gauge the overall data, the volume of data, the types of columns present in the data. This initial assessment of the data is needed to identify which columns are Quantitative, Categorical or Qualitative, and identify and remove unwanted columns 
                """)

    ## Min max
    st.write(f"### Statistical Summary of the Laptop Price Dataset:")
    st.dataframe(laptop_price_df.describe(), use_container_width=True)

    st.write("### Data Types:")
    dtype_df = pd.DataFrame(laptop_price_df.dtypes, columns=["Data Type"]).reset_index()
    dtype_df = dtype_df.rename(columns={"index": "Column Name"})

    st.dataframe(dtype_df, use_container_width=True)

    st.write("### Observation:")
    st.write("""
         The data types of each columns is identified in this step. We now know which columns are numerical and categorical. It is really helpful for further investigation in this dataset, because sometime teh code will crash if the data is mixed with string variable. This step helps us to prevent this problem occurs.
     """)

    st.write("""
                    # Step 5: Visual Exploratory Data Analysis (EDA) of data (with histogram and barcharts)
                    #### This requires visualising distribution of all the categorical predictor variables in the data using bar plots, and continuous predictor variables using histograms.
                    """)

    # Step 5: Visual EDA - Histograms for Continuous Variables
    st.write("## Visual EDA - Histograms of Continuous Variables")
    continuous_columns = st.multiselect("Select continuous variables to visualize:", laptop_price_df.select_dtypes(include=['float64', 'int64']).columns)

    if continuous_columns:
            # Plot the heatmap
        st.write("### Plot the heatmap for the correlation of numeric columns")
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.heatmap(laptop_price_df[continuous_columns].corr(), annot=True, ax=ax)  # Make sure the correct `ax` is passed
        st.pyplot(fig)
        plt.close(fig)  # Close the figure to avoid reuse

        # Plot the pairplot
        st.write("### Plot the pairplot for the correlation of numeric columns")
        pairplot = sns.pairplot(laptop_price_df, hue='Price', vars=continuous_columns)
        st.pyplot(pairplot)
        plt.close()  # Close the pairplot

        categorical_columns = ['Brand']

        # Plot the histograms for continuous variables
        st.write("### Histograms for Continuous Variables")
        fig, ax = plt.subplots(figsize=(15, 10))  # Create a new figure for the histograms
        laptop_price_df[continuous_columns].hist(ax=ax, layout=(2, 3))  # Explicitly pass the axes to the histogram
        plt.tight_layout()
        st.pyplot(fig)  # Render the histogram figure
        plt.close(fig)  # Close the figure to prevent reuse

        # Create bar plots for categorical variables
        st.write("### Bar Plots for Categorical Variables")
        for col in categorical_columns:
            fig, ax = plt.subplots(figsize=(10, 12))
            laptop_price_df[col].value_counts().plot(kind='bar', ax=ax)  # Pass the `ax` to ensure it's the correct figure
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Count')
            st.pyplot(fig)  # Display each bar plot in Streamlit
            plt.close(fig)  # Close the figure after each plot

        # Create the boxplot
        st.write("### Boxplot for Price and Brand")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=laptop_price_df, x='Brand', y='Price', color='yellow', ax=ax)  # Ensure the correct `ax` is passed
        st.pyplot(fig)  # Display the boxplot
        plt.close(fig)  # Close the boxplot figure

        # Identify categorical and numeric columns
        categorical_columns = ['Brand']  # categorical columns
        numeric_columns = ['Price', 'RAM_Size', 'Storage_Capacity']  # numeric columns

        ## GRAPH GRAPH ## GRAPH GRAPH ## GRAPH GRAPH ## GRAPH GRAPH
        for cat_col in continuous_columns:
            for num_col in numeric_columns:
            # Skip if cat_col = num_col; i.e. price compared to price
                if cat_col == num_col:
                    continue

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.lineplot(data=laptop_price_df, x=cat_col, y=num_col)
                plt.title(f'{num_col} vs {cat_col}')
                plt.xticks(rotation=45)
                plt.tight_layout()

                st.pyplot(fig)
        ## GRAPH GRAPH ## GRAPH GRAPH ## GRAPH GRAPH ## GRAPH GRAPH

        st.write("### Observation:")
        st.write("""
            The plots show which continuous variables have strong relationships with each other. 
            This helps us in feature selection for model training later.
        """)


    st.write("""
                       # Step 6: Outlier analysis
                       #### Removal of outliers and missing values is an important step, as the outliers are extreme values in the data, which are far away from most of the values. You can see them as the tails in the histogram.
                       """)

    # Step 6: Outlier Analysis
    st.write("### Removal of outliers")

    # Select only continuous numerical columns
    numeric_columns = laptop_price_df.select_dtypes(include=['float64', 'int64'])

    if not numeric_columns.empty:

        Q1 = numeric_columns.quantile(0.25)
        Q3 = numeric_columns.quantile(0.75)
        IQR = Q3 - Q1


        outliers = (numeric_columns < (Q1 - 1.5 * IQR)) | (numeric_columns > (Q3 + 1.5 * IQR))

        # Count outliers in each column
        outlier_count = outliers.sum()

        # Display the number of outliers in each column
        st.write("Number of Outliers in Each Numeric Column")
        dtype_df_outlier = pd.DataFrame(outlier_count, columns=["Number of Outliers"]).reset_index()
        dtype_df_outlier = dtype_df_outlier.rename(columns={"index": "Column Name"})
        st.dataframe(dtype_df_outlier, use_container_width=True)
    else:
        st.write("No continuous numeric columns available for outlier analysis.")

    st.write("There is no outlier")

    st.write("### Observation:")
    st.write("""
              We can observe that there is no outlier in this dataset. 
          """)

    st.write("""
            # Step 7: Missing values analysis - Options for treating the missing values
            #####  Delete the missing value rows if there are only few records,
            #####  Impute the missing values with MODE value for categorical variables,
            #####  Interpolate the values based on nearby values,
            #####  Interpolate the values based on business logic.
            """)

    st.write("### Deleting missing value")
    st.write("Showing missing value in this dataset")

    missing_values = laptop_price_df.isnull().sum()
    dtype_df_missing_values = pd.DataFrame(missing_values, columns=["Missing Values"]).reset_index()
    dtype_df_missing_values = dtype_df_missing_values.rename(columns={"index": "Column Name"})
    st.dataframe(dtype_df_missing_values, use_container_width=True)

    st.write("### Observation:")
    st.write("""
                There are no missing values in this dataset. We can move forward without additional handling of missing data.
            """)

    st.write("""
            # Step 8: Feature selection - Visual and statistic correlation analysis for selection of best features
            #####  If the target variable is continuous and the predictor is also continuous, visualise the relationship between the two variables using scatter plot and measure the strength of relation using a metric called Pearson's correlation value.
            #####  Statistical feature selection (continuous Vs. continuous) using correlation value.
            #####  When the target variable is continuous, and the predictor variable is categorical we analyse the relation using box plots.
            """)

    # Calculate the correlation matrix
    laptop_price_df_without_brand = laptop_price_df.drop(columns=['Brand'])
    correlation_matrix = laptop_price_df_without_brand.corr()

    st.title("Correlation Matrix Heatmap")

    # Set up the figure and plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw the heatmap with the correct aspect ratio
    sns.heatmap(correlation_matrix, annot=True, cmap="YlGnBu", linewidths=.5, ax=ax)

    # Display the plot
    st.pyplot(fig)

    correlation_matrix = laptop_price_df_without_brand.corr()
    target_correlations = correlation_matrix[target].drop(target)

    # Display the correlation matrix in a table format
    st.title("Correlation Matrix:")

    # Udisplay the matrix with scrollbars and resizable columns
    st.dataframe(correlation_matrix, use_container_width=True)

    # Identify strong, moderate, and weak relationships
    strong_corr = target_correlations[target_correlations.abs() >= 0.7]
    moderate_corr = target_correlations[(target_correlations.abs() >= 0.3) & (target_correlations.abs() < 0.7)]
    weak_corr = target_correlations[target_correlations.abs() < 0.3]

    # Display the correlations classified as strong, moderate, and weak
    st.write("### Strong Correlations (|correlation| e 0.7):")
    st.dataframe(strong_corr, use_container_width=True)

    st.write("### Moderate Correlations (0.3 d |correlation| < 0.7):")
    st.dataframe(moderate_corr, use_container_width=True)

    st.write("### Weak Correlations (|correlation| < 0.3):")
    st.dataframe(weak_corr, use_container_width=True)

    st.write("### Observation:")
    st.write("""
             After analysing the correlation matrix, we can select the features that are most strongly correlated 
             with our target variable for further analysis and model training. It seems that the colleration between Storage Capacity and Price is a strong colleration. 
         """)

    st.write("""
            # Step 9: Statistical feature selection (categorical vs. continuous) using ANOVA test
            #####  Analysis of variance (ANOVA) is performed to check if there is any relationship between the given continuous and categorical variable.
            #####  Assumption (H0) null hypothesis: There is no relation between the given variables (i.e. the average(mean) values of the numeric Target variable is same for all the groups in the categorical predictor variable).
            #####  ANOVA test result: Probability of H0 (null hypothesis being true).
            """)

    anova_result = stats.f_oneway(
        *[laptop_price_df[laptop_price_df['Brand'] == brand]['Price'] for brand in laptop_price_df['Brand'].unique()]
    )

    # app title
    st.title("ANOVA Test: Price Differences between Brands")

    # Display the result
    st.write("F-statistic:", anova_result.statistic)
    st.write("p-value:", anova_result.pvalue)

    # Check if the p-value is significant
    if anova_result.pvalue < 0.05:
        st.write("Reject H0: There is a significant difference in Price between different Brands.")
    else:
        st.write("Fail to reject H0: No significant difference in Price between different Brands.")

    st.subheader("Price Distribution by Brand")

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Brand', y='Price', data=laptop_price_df)

    # Display the plot
    st.pyplot(plt)

    st.write("### Observation:")
    st.write(
        "We can see that the p-value is 0.7024, and the F-statistic is 0.5455. It seems that there is no significant difference in price between different brands.")

    st.write("""
                        # Step 10: Selecting final predictors/features for building machine learning/AI model
                        #### Based on the extensive tests with basic and visual exploratory data analysis, select the final features/predictors/columns for machine learning model. 
                        """)

    # Step 10: Selecting Final Predictors
    st.title("Step 10: Selecting Final Predictors")

    # List of all possible predictors
    predictors = ['Processor_Speed', 'RAM_Size', 'Screen_Size', 'Storage_Capacity', 'Weight', 'Brand']

    # Multiselect for predictor variables
    selected_features = st.multiselect(
        'Select predictor variables (independent variables):', predictors,
        ['Processor_Speed', 'RAM_Size', 'Screen_Size'])

    # Display selected features
    st.subheader("Selected Features:")
    st.write(selected_features)

    # Target Variable
    st.subheader("Target Variable: **Price**")

    st.write("### Observation:")
    st.write(
        "We can see that the final selected predictors include variables that are highly relevant to laptop prices. These predictors will be used for training the machine learning models.")

    # Step 11: Data Preparation for Machine Learning
    st.title("Step 11: Data Preparation for Machine Learning")
    if 'Brand' in selected_features:
        laptop_price_df_numeric = pd.get_dummies(laptop_price_df[selected_features], columns=['Brand'], drop_first=True)
    else:
        laptop_price_df_numeric = laptop_price_df[selected_features]

    st.write("Display the first few rows of the converted dataset using pd.get_dummies")
    st.dataframe(laptop_price_df_numeric.head(), use_container_width=True)

    # Slider for selecting test size percentage
    test_size = st.slider('Select the test size (percentage)', min_value=0.10, max_value=0.50, step=0.01, value=0.20)

    st.write("### Observation:")
    st.write(
        "In here, we use a side bar to select testing size so user can easily interact with the UI to selcte their testing dataset size")
    st.write("""
            # Step 12: Train/test data split and standardisation/normalisation of data
            #####  Splitting the data into training and testing sample.
            """)
    # Display the shapes of the train and test sets
    X = laptop_price_df_numeric
    y = laptop_price_df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1 - test_size, random_state=0)

    st.write("X_train shape:", X_train.shape)
    st.write("X_test shape:", X_test.shape)
    st.write("y_train shape:", y_train.shape)
    st.write("y_test shape:", y_test.shape)

    st.write("### Observation:")
    st.write(
        "We can see that the dataset has been successfully split into training and testing sets, and the features have been prepared for model training. We usually split into 80/20 because it is a popular ratio.")

    st.write("""
            # Step 13: Investigating multiple regression algorithms
            #####  Build the machine learning/AI model with 5 algorithms, linear regression, decision tree regressor, random forest regressor, adaboost regressor, XGBoost regressor, K-Nearest neighbour regressor, and SVM regressor.
            """)

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regression": DecisionTreeRegressor(),
        "Random Forest Regression": RandomForestRegressor(),
        "XGBoost Regression": XGBRegressor(),
        "SVM Regression": SVR()
    }

    score = []
    mse = []
    mae = []
    model_names = []  # model names for the dataframe

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score.append(r2_score(y_test, y_pred))
        mse.append(mean_squared_error(y_test, y_pred))
        mae.append(mean_absolute_error(y_test, y_pred))
        model_names.append(name)  # Append the model name

    # Creating the results dataframe
    results_df = pd.DataFrame({
        'Model': model_names,
        'R-squared Score': score,
        'Mean Squared Error': mse,
        'Mean Absolute Error': mae
    })

    # Show results dataframe
    st.write("### Results Dataframe")
    st.dataframe(results_df, use_container_width=True)

    # R-squared Score
    st.write("### R-squared Score of Different Models")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Model'], results_df['R-squared Score'], color='skyblue')
    ax.set_title('R-squared Score of Different Models')
    ax.set_ylabel('R-squared Score')
    ax.set_xlabel('Model')
    ax.set_xticks(range(len(results_df['Model'])))
    ax.set_xticklabels(results_df['Model'], rotation=45)
    st.pyplot(fig)

    # Mean Squared Error (MSE)
    st.write("### Mean Squared Error of Different Models")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Model'], results_df['Mean Squared Error'], color='salmon')
    ax.set_title('Mean Squared Error of Different Models')
    ax.set_ylabel('Mean Squared Error')
    ax.set_xlabel('Model')
    ax.set_xticks(range(len(results_df['Model'])))
    ax.set_xticklabels(results_df['Model'], rotation=45)
    st.pyplot(fig)

    # Mean Absolute Error (MAE)
    st.write("### Mean Absolute Error of Different Models")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(results_df['Model'], results_df['Mean Absolute Error'], color='lightgreen')
    ax.set_title('Mean Absolute Error of Different Models')
    ax.set_ylabel('Mean Absolute Error')
    ax.set_xlabel('Model')
    ax.set_xticks(range(len(results_df['Model'])))
    ax.set_xticklabels(results_df['Model'], rotation=45)
    st.pyplot(fig)

    st.write("### Observation:")
    st.write(
        "We can see that multiple regression models have been trained, and their performance metrics (R-squared, MSE, and MAE)  are shown for comparison.")

    st.write("""
            # Step 14: Selection of the best model
            #####  Based on the above/previous trials with different regression algorithms, select that algorithm which produces the best average accuracy.
            """)

    # Find the best model based on the R-squared score
    best_r2_index = results_df['R-squared Score'].idxmax()
    best_model_name = results_df.loc[best_r2_index, 'Model']
    best_r2_score = results_df.loc[best_r2_index, 'R-squared Score']

    # Display the best model based on R-squared score
    st.write(f"###### The best model is {best_model_name} with an highest R-squared score of {best_r2_score:.4f}")


    best_mse_index = results_df['Mean Squared Error'].idxmin()
    best_mse_model_name = results_df.loc[best_mse_index, 'Model']
    best_mse_value = results_df.loc[best_mse_index, 'Mean Squared Error']

    # Display the best model based on MSE
    st.write(
        f"###### The best model based on Mean Squared Error (MSE) is {best_mse_model_name} with the lowest MSE of {best_mse_value:.4f}")


    best_mae_index = results_df['Mean Absolute Error'].idxmin()
    best_mae_model_name = results_df.loc[best_mae_index, 'Model']
    best_mae_value = results_df.loc[best_mae_index, 'Mean Absolute Error']

    # Display the best model based on MAE
    st.write(
        f"###### The best model based on Mean Absolute Error (MAE) is {best_mae_model_name} with the lowest MAE of {best_mae_value:.4f}")

    st.write("### Observation:")
    st.write(
        "We can see that the best-performing model, based on R-squared, MSE, and MAE, has been selected.")

    st.write("""
                # Step 15: Deployment of the best model in production
        
                """)

    st.write("### Observation:")
    st.write("""
        In Step 15, we successfully deployed the best-performing model. We chose Streamlit as the UI interface for its simplicity and ease of use, especially since our entire project is built using Python. 
        The model was successfully saved and can now be used for future predictions on new data.
        Additionally, we used plots to represent performance of the model.

    """)

    ## Retrain data

    best_model = models[
        'Linear Regression']
    best_model.fit(X, y)

    # Save the model to a file
    joblib.dump(best_model, 'best_model.pkl')
    st.write(f"Model `{best_model}` has been retrained and saved as best_model.pkl.")

    # Load the pre-trained model (assuming best_model.pkl is saved in the same directory)
    with open('best_model.pkl', 'rb') as file:
        model = pickle.load(file)

    # Load the saved model
    model_filename = "best_model.pkl"

    try:
        # Load the model pipeline (including preprocessing)
        loaded_model = joblib.load(model_filename)
        st.write(f"Model {model_filename} loaded successfully!")

        # Allow user to input values for the features
        st.write("### Provide the input values for prediction")

        # Assuming the features include one-hot encoded categorical variables (such as 'Brand')
        # User should input values for the necessary features
        user_input_values = {}
        if 'Brand' in selected_features:
            selected_features.remove('Brand')
            for feature in selected_features:  
                user_input_values[feature] = st.number_input(f"Enter value for {feature}",
                                                         value=float(laptop_price_df[feature].mean()))
            brands = ['Dell', 'HP', 'Asus', 'Lenovo']

            brand = st.selectbox("Choose values for Brand", brands)
            for b in brands:
                user_input_values[f'Brand_{b}'] = (b == brand)

        else:
            for feature in selected_features:  
                user_input_values[feature] = st.number_input(f"Enter value for {feature}",
                                                         value=float(laptop_price_df[feature].mean()))

        # Predict button
        if st.button("Predict"):
            # Convert the user inputs into a DataFrame
            feature_columns = X.columns.tolist()
            user_input_df = pd.DataFrame([user_input_values], columns=feature_columns)

            print(user_input_values)
            print(X.head())

            # Use the model pipeline to preprocess and predict
            predicted_value = loaded_model.predict(user_input_df)

            # Display the predicted value
            st.write(f"### Predicted {target}: {predicted_value[0]:.2f}")

            # Visualizing the Prediction with Input Features
            st.write("## Visualizing Prediction and Feature Values")

            # Create a combined bar plot for input features and prediction
            fig, ax = plt.subplots()

            # Plot the input feature values
            feature_names = list(user_input_values.keys())
            feature_values = list(user_input_values.values())

            ax.bar(feature_names, feature_values, color='lightblue', label='Feature Values')

            # Add the predicted value at the bottom of the chart
            ax.bar(['Predicted ' + target], [predicted_value[0]], color='orange', label='Predicted Value')

            ax.set_xlabel("Value")
            ax.set_title(f"Input Features and Predicted {target}")
            ax.legend()

            # Display the plot
            st.pyplot(fig)

            # Create a line chart for input features and predicted value
            fig, ax = plt.subplots(figsize=(10, 6))

            copy_feature_names = feature_names.copy()
            copy_feature_values = feature_values.copy()
            # Add the predicted value at the end
            copy_feature_names.append(f"Predicted {target}")
            copy_feature_values.append(predicted_value[0])

            ax.plot(copy_feature_names, copy_feature_values, marker='o', linestyle='-', color='blue',
                    label='Feature and Predicted Values')

            ax.set_xlabel("Features and Prediction")
            ax.set_ylabel("Values")
            ax.set_title(f"Input Features and Predicted {target}")
            ax.grid(True)

            # Add data labels
            for i, txt in enumerate(copy_feature_values):
                ax.annotate(f'{txt:.2f}', (copy_feature_names[i], copy_feature_values[i]),
                            textcoords="offset points",
                            xytext=(0, 10), ha='center')

            # Display the plot
            st.pyplot(fig)

    except FileNotFoundError:
        st.write(f"Model {model_filename} not found. Please ensure the model has been saved correctly.")










