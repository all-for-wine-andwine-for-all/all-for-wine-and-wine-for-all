#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import wrangle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import statsmodels.api as sm
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import seaborn as sns
import scipy.stats as stats
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
# In[ ]:


def remove_outliers_iqr(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Loop through numeric columns
    for col in num_cols:
        # Calculate the IQR of the column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the upper and lower bounds for outliers
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR

        # Filter out the outliers from the DataFrame
        df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df

def remove_outliers_iqr_loop(df, multiplier=1.5):
    # Select only numeric columns
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    
    # Initialize the number of outliers
    num_outliers = 1
    
    while num_outliers > 0:
        # Initialize the number of outliers
        num_outliers = 0
        
        # Loop through numeric columns
        for col in num_cols:
            # Calculate the IQR of the column
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # Define the upper and lower bounds for outliers
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR

            # Filter out the outliers from the DataFrame
            num_outliers += df[col][(df[col] < lower_bound) | (df[col] > upper_bound)].count()
            df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
    
    return df



def wrangle_zillow_data():
    df = get_zillow_data()
    df = drop_all_null(df)
    df = clean_zillow_data(df)
    df = prepare_data(df)
    #df = one_hot_encode_data(df)
    df = rename_encoded_columns(df)
    #df = scale_data(df)
    #df = scale_y_data(df)
    #df = clean_up_X_y(df)
    df = remove_outliers_iqr_loop(df, multiplier=1.5)
    df = split_data(df)
    summarize_data(df)
    #df = drop_missing_data(df, 0.9)
   
    return df, custom_desc_df

def acquire_wrangle_data():
    df = acquire_zillow_data()
    #df = clean_zillow_data(df)
    df = handle_missing_values(df, prop_required_column, prop_required_row)
    
    custom_desc_df = custom_describe(df)
    df = custom_describe(df)
    #X, y = prepare_data(df)
    #df = pd.concat([X, y], axis=1)
    df = scale_data(df)
    df = scale_y_data(df)
    
    return df




def make_sample(df, frac=0.003):
    """
    This function takes a DataFrame as input and returns a random sample of the data.
    
    Parameters
    ----------
    df : DataFrame
        The input DataFrame to take a random sample from.
    frac : float, optional
        The fraction of the data to take as a sample. Default is 0.001 (0.1%).
        
    Returns
    -------
    DataFrame
        The random sample of the input DataFrame.
    """
    sample_df = df.sample(frac=frac)
    return sample_df

# Example usage
# sample_df = make_sample(df, frac=0.001)


def count_outliers_iqr(df):
    q1 = df.quantile(0.25)
    q3 = df.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outliers = ((df < lower_bound) | (df > upper_bound)).sum().sum()
    return outliers
    
    
        
def get_stats(df, target_col):
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    X = sm.add_constant(X)  # Add a constant term for the intercept
    OLS = sm.OLS(y, X).fit()
    print(OLS.summary())
    return OLS



    for name, model in models:
        mse = -np.mean(cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5))
        r2 = np.mean(cross_val_score(model, X, y, scoring='r2', cv=5))
        print(f"{name}: Mean Squared Error = {mse:.4f}, R^2 Score = {r2:.4f}")
        
        
def model_eval(df, target_column):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [col for col in df.columns if col != target_column]

    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Separate the features and target for each dataset
    X_train, y_train = train[feature_columns], train[target_column]
    X_validate, y_validate = validate[feature_columns], validate[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_validate, y_validate)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

   # for feature in feature_columns:
        #sns.regplot(x=feature, y=target_column, data=df)
        #plt.title(f"{feature} vs. {target_column}")
        #plt.show()

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")
        
def model_evals(df_train, target_column, df_validation):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scipy.stats as stats
    from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.ensemble import RandomForestRegressor

    feature_columns = [col for col in df_validation.columns if col != target_column]

    # Use the validation set
    X_val, y_val = df_validation[feature_columns], df_validation[target_column]

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_val, y_val)
        score = model.score(X_val, y_val)  # Evaluate the model on the validation set
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    for feature in feature_columns:
        sns.regplot(x=feature, y=target_column, data=df_validation)
        plt.title(f"{feature} vs. {target_column}")
        plt.show()

        
def model_interpretation(df, target_column, best_params):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor

    
    best_params = {
    'n_estimators': 200,
    'max_depth': 30,
    'min_samples_split': 2,
    'min_samples_leaf': 1,
    'bootstrap': True
    }
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(**best_params)

    rf.fit(X_train, y_train)

    importances = rf.feature_importances_

    feature_importances = pd.DataFrame({'feature': X.columns, 'importance': importances})

    feature_importances = feature_importances.sort_values(by='importance', ascending=False)

    print(feature_importances)


def feature_importance_bar_chart(feature_importances):

    plt.figure(figsize=(10, 5))
    plt.bar(feature_importances['feature'], feature_importances['importance'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    plt.title('Feature Importances')
    plt.xticks(rotation=45)
    plt.show()
    
def model_validation(df, target_column):
    feature_columns = [col for col in df.columns if col != target_column]

    X_train, X_test, y_train, y_test = train_test_split(df[feature_columns], df[target_column], test_size=0.3, random_state=42)

    models = [
        ('Linear Regression', LinearRegression()),
        ('Lasso', Lasso()),
        ('Ridge', Ridge()),
        ('ElasticNet', ElasticNet()),
        ('Decision Tree', DecisionTreeRegressor()),
        ('Random Forest', RandomForestRegressor())
    ]

    def get_p_values(X, y, model):
        n = X.shape[0]
        p = X.shape[1]
        y_pred = model.predict(X)
        residuals = y - y_pred
        RSS = np.sum(residuals**2)
        MSE = RSS / (n - p - 1)

        se_beta = np.sqrt(np.diagonal(MSE * np.linalg.inv(np.dot(X.T, X))))
        t_stat = model.coef_ / se_beta
        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stat), df=n - p - 1))

        return p_values

    best_model = None
    best_score = -np.inf

    for name, model in models:
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        if score > best_score:
            best_score = score
            best_model = (name, model)

    print(f"Best Model: {best_model[0]}, R^2 score: {best_score:.3f}")

    p_values = get_p_values(X_train, y_train, best_model[1])
    for feature, p_value in zip(feature_columns, p_values):
        print(f"P>|t| for {feature}: {p_value:.3f}")
        
        
def feature_correlation(df):
    # Remove 'assessed_property_value' column from the list of features
    features = [column for column in df.columns if column != 'target']

    # Set the significance level
    alpha = 0.05

    # Loop through all features and perform hypothesis test
    for feature in features:
        X = df[feature]
        y = df['target']

        # Calculate Pearson correlation coefficient and p-value
        correlation_coefficient, p_value = stats.pearsonr(X, y)
        correlation_coefficient_s, p_value_s = stats.spearmanr(X, y)

        # Print the results
        print(f"Feature: {feature}")
        print(f"Pearson correlation coefficient: {correlation_coefficient}")
        print(f"P-value: {p_value}")
        
        print("\n")
        print(f"Spearman correlation coefficient: {correlation_coefficient_s}")
        print(f"P-value: {p_value_s}")

        # Test the null hypothesis
        if p_value < alpha:
            print("\033[32mReject the null hypothesis (H0): There is a linear relationship'.\033[0m")
        else:
            print("\033[31mFail to reject the null hypothesis (H0): There is no evidence of a linear relationship'.\033[0m")
        
        print("\n")
        
        
def data_dict():
    data_dict = {
        'id': 'Unique identifier for each property',
        'parcelid': 'Unique identifier for each property, used in conjunction with "assessmentyear" to form a composite primary key',
        'bedroomcnt': 'Number of bedrooms in the property',
        'bathroomcnt': 'Number of bathrooms in the property',
        'fireplacecnt': 'Number of fireplaces in the property',
        'calculatedbathnbr': 'Number of bathrooms in the property (including fractional bathrooms)',
        'calculatedfinishedsquarefeet': 'Total finished living area of the property, in square feet',
        'fullbathcnt': 'Number of full bathrooms in the property (including fractional bathrooms)',
        'garagecarcnt': 'Number of cars that can fit in the garage, if applicable',
        'garagetotalsqft': 'Total square footage of the garage, if applicable',
        'latitude': 'Latitude of the property',
        'longitude': 'Longitude of the property',
        'lotsizesquarefeet': 'Total area of the lot, in square feet',
        'regionidzip': 'Zip code of the property',
        'taxvaluedollarcnt': 'Total tax assessed value of the property, in dollars',
        'roomcnt': 'Total number of rooms in the property (including bedrooms and bathrooms)',
        'yearbuilt': 'Year the property was built',
        'numberofstories': 'Number of stories in the property, if applicable',
        'assessmentyear': 'Year of the property assessment, used in conjunction with "parcelid" to form a composite primary key',
        'landtaxvaluedollarcnt': 'Tax assessed value of the land, in dollars',
        'structuretaxvaluedollarcnt': 'Tax assessed value of the structure, in dollars',
        'taxamount': 'Total property tax for the assessment year, in dollars'
    }

    data_dict_df = pd.DataFrame.from_dict(data_dict, orient='index', columns=['Description'])
    return data_dict_df


def split_and_evaluate_ols(df, target_column):
    from sklearn.model_selection import train_test_split

    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    # Evaluate the OLS model
    wrangle.evaluate_ols(train, validate, test, target_column)
    
    
def evaluate_ols_with_splits(df, target_column):
    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

def evaluate_ols(train, validate, test, target_column):
    feature_columns = [col for col in train.columns if col != target_column]

    X_train, y_train = train[feature_columns], train[target_column]
    X_validate, y_validate = validate[feature_columns], validate[target_column]
    X_test, y_test = test[feature_columns], test[target_column]

    model = LinearRegression()
    model.fit(X_train, y_train)

    def evaluate_set(name, X, y):
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mse = mean_squared_error(y, y_pred)
        print(f"{name} R^2 Score: {r2:.3f}")
        print(f"{name} Mean Squared Error: {mse:.3f}")
        return r2

    train_r2 = evaluate_set("Train", X_train, y_train)
    validate_r2 = evaluate_set("Validation", X_validate, y_validate)
    test_r2 = evaluate_set("Test", X_test, y_test)

    print(f"Difference in R^2 Scores:")
    print(f"Train-Validation: {abs(train_r2 - validate_r2):.3f}")
    print(f"Train-Test: {abs(train_r2 - test_r2):.3f}")
    print(f"Validation-Test: {abs(validate_r2 - test_r2):.3f}")
    
def final_split_data(df, target_column):
    # Split the data into train (70%), validation (15%), and test (15%) sets
    train, temp = train_test_split(df, test_size=0.3, random_state=42)
    validate, test = train_test_split(temp, test_size=0.5, random_state=42)

    return train, validate, test

def custom_describe(df):
    desc = df.describe(include='all').T

    # Adding additional columns
    desc['count_nulls'] = df.isnull().sum()
    desc['pct_nulls'] = (desc['count_nulls'] / len(df)) * 100
    desc['num_rows_missing'] = len(df) - desc['count']
    desc['pct_rows_missing'] = (desc['num_rows_missing'] / len(df)) * 100
    desc['dtype'] = df.dtypes

    # Add results from sub-functions to the desc DataFrame
    desc['distribution_type'] = df.apply(distribution_type)
    desc['skewness'] = df.apply(column_skewness)
    desc['skew_type'] = df.apply(skew_type)
    desc['data_type'] = df.apply(data_type)
    desc['num_outliers'] = df.apply(iqr_outliers)
    desc['variable_type'] = df.apply(variable_type)

    # Calculate correlations between numeric columns
    correlations = df.corr().abs().unstack().sort_values(ascending=False).drop_duplicates()
    correlations = correlations[(correlations != 1) &                     correlations.index.get_level_values(0).equals(correlations.index.get_level_values(1))]

    
    for col in df.columns:
        print("\nColumn:", col)
        
        if df[col].dtype != 'O':
            sns.histplot(df[col], kde=True)
            plt.title(f'Distribution of {col}')
            plt.show()

    # Reorder columns
    columns_order = [
        'num_rows_missing', 'pct_rows_missing','count', 'count_nulls', 'pct_nulls', 'mean',                   'std','min', '25%', '50%', '75%', 'max', 'unique', 'top', 'freq', 'dtype',                             'distribution_type', 'skewness', 'skew_type', 'data_type', 'num_outliers', 'variable_type'
    ]
    desc = desc[columns_order]

    # Display the custom describe DataFrame with left-aligned column names
    display(desc.style.set_properties(**{'text-align': 'left'}))


# Sub-functions go here (dist#ribution_type, column_skewness, skew_type, data_type, iqr_outliers, variable_type)

# Load your data into a DataFrame called 'df'
# For example, you can use he following line to load a CSV file
# df = pd.read_csv('your_data_file.csv')

# Call the custom_describe function
# custom_desc_df = custom_describe(df)


def column_skewness(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    return stats.skew(column.dropna())

def skew_type(column):
    skew = column_skewness(column)
    if skew is None:
        return None
    if skew > 0:
        return 'Right'
    elif skew < 0:
        return 'Left'
    else:
        return 'Symmetric'

def data_type(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    unique_count = column.nunique()
    if unique_count / len(column) < 0.05:
        return 'Discrete'
    else:
        return 'Continuous'
    
def iqr_outliers(column):
    if not np.issubdtype(column.dtype, np.number):
        return None
    
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return ((column < lower_bound) | (column > upper_bound)).sum() 
    
def variable_type(column):
    num_unique_values = len(column.unique())
    
    if column.dtype == np.object:
        if num_unique_values <= 10:
            return 'nominal'
        else:
            return 'categorical'
    elif column.dtype == np.int64 or column.dtype == np.float64:
        if num_unique_values <= 10:
            return 'ordinal'
        else:
            return 'numerical'
    else:
        return 'unknown'
