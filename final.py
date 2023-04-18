# =======================================================================================================
# Table of Contents START
# =======================================================================================================

'''
1. Orientation
2. Imports
3. visual1
4. stat1
5. visual2
6. stat2
7. visual3
8. stat3
9. visual4
10. stat4
11. baseline
12. models
13. best_model
14. acquire
15. prepare
16. visual5
'''

# =======================================================================================================
# Table of Contents END
# Table of Contents TO Orientation
# Orientation START
# =======================================================================================================

'''
The purpose of this file is to create functions in order to expedite and maintain cleanliness
of the final_report.ipynb
'''

# =======================================================================================================
# Orientation END
# Orientation TO Imports
# Imports START
# =======================================================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# =======================================================================================================
# Imports END
# Imports TO visual1
# visual1 START
# =======================================================================================================

def visual1():
    '''
    Returns the 1st specific visual for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    regplot of sulphates and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    sns.regplot(data=wines, x='sulphates', y='quality', line_kws={'color':'red'})
    plt.title('Sulphates vs. Wine Quality')
    plt.show()


# =======================================================================================================
# visual1 END
# visual1 TO stat1
# stat1 START
# =======================================================================================================

def stat1():
    '''
    Returns the 1st specific stat for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    stats of sulphates and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    r, p =stats.spearmanr(wines.sulphates, wines.quality)
    print('\033[32m========== REJECT NULL HYPOTHESIS! ==========')
    print('\033[35mFeatures:\033[0m Sulphates vs. Quality')
    print(f'\033[35mCorrelation:\033[0m {r:.4f}')
    print(f'\033[35mP-Value:\033[0m {p:.4f}')

# =======================================================================================================
# stat1 END
# stat1 TO visual2
# visual2 START
# =======================================================================================================

def visual2():
    '''
    Returns the 2nd specific visual for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    regplot of alcohol and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    sns.regplot(data=wines, x='alcohol', y='quality', line_kws={'color':'red'})
    plt.title('Alcohol vs. Wine Quality')
    plt.show()

# =======================================================================================================
# visual2 END
# visual2 TO stat2
# stat2 START
# =======================================================================================================

def stat2():
    '''
    Returns the 2nd specific stat for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    stats of alcohol and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    r, p =stats.spearmanr(wines.alcohol, wines.quality)
    print('\033[32m========== REJECT NULL HYPOTHESIS! ==========')
    print('\033[35mFeatures:\033[0m Alcohol vs. Quality')
    print(f'\033[35mCorrelation:\033[0m {r:.4f}')
    print(f'\033[35mP-Value:\033[0m {p:.4f}')

# =======================================================================================================
# stat2 END
# stat2 TO visual3
# visual3 START
# =======================================================================================================

def visual3():
    '''
    Returns the 3rd specific visual for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    regplot of residual sugars and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    sns.regplot(data=wines, x='residual sugar', y='quality', line_kws={'color':'red'})
    plt.title('Residual Sugars vs. Wine Quality')
    plt.show()

# =======================================================================================================
# visual3 END
# visual3 TO stat3
# stat3 START
# =======================================================================================================

def stat3():
    '''
    Returns the 3rd specific stat for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    stats of residual sugar and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    r, p =stats.spearmanr(wines['residual sugar'], wines.quality)
    print('\033[31m========== ACCEPT NULL HYPOTHESIS! ==========')
    print('\033[35mFeatures:\033[0m Residual Sugar vs. Quality')
    print(f'\033[35mCorrelation:\033[0m {r:.4f}')
    print(f'\033[35mP-Value:\033[0m {p:.4f}')

# =======================================================================================================
# stat3 END
# stat3 TO visual4
# visual4 START
# =======================================================================================================

def visual4():
    '''
    Returns the 4th specific visual for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    regplot of volatile acidity and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    sns.regplot(data=wines, x='volatile acidity', y='quality', line_kws={'color':'red'})
    plt.title('Volatile Acidity vs. Wine Quality')
    plt.show()

# =======================================================================================================
# visual4 END
# visual4 TO stat4
# stat4 START
# =======================================================================================================

def stat4():
    '''
    Returns the 4th specific stat for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    stats of volatile acidity and quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    r, p =stats.spearmanr(wines['volatile acidity'], wines.quality)
    print('\033[32m========== REJECT NULL HYPOTHESIS! ==========')
    print('\033[35mFeatures:\033[0m Volatile Acidity vs. Quality')
    print(f'\033[35mCorrelation:\033[0m {r:.4f}')
    print(f'\033[35mP-Value:\033[0m {p:.4f}')

# =======================================================================================================
# stat4 END
# stat4 TO baseline
# baseline START
# =======================================================================================================

def baseline():
    '''
    Returns the baseline accuracy score for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    baseline accuracy score of the wine quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    baseline = round((wines.quality == wines.quality.mode()[0]).sum() / wines.shape[0], 3)
    return baseline

# =======================================================================================================
# baseline END
# baseline TO models
# models START
# =======================================================================================================

def models():
    '''
    Returns a pandas dataframe with the baseline and 4 of the best models within the group

    INPUT:
    NONE

    OUTPUT:
    A pandas dataframe with model scores
    '''
    models_dict = {
    'type' : [
        'baseline',
        'Random Forest Classifier1',
        'Random Forest Classifier2',
        'Random Forest Classifier3',
        'Random Forest Classifier4'
    ],
    'hyperparameters' : [
        'None',
        'depth=3, ccpalpha=0.0015',
        'depth=5',
        'depth=5, ccpalpha=0.0007',
        'n_estimators=100'
    ],
    'clusters' : [
        'None',
        'None',
        'Yes',
        'Yes',
        'None'
    ],
    'train_accuracy' : [
        0.44,
        0.55,
        0.52,
        0.56,
        1.00
    ],
    'validate_accuracy' : [
        0.44,
        0.54,
        0.40,
        0.54,
        0.88
    ],
    'diff_accuracy' : [
        0.000,
        -0.01,
        -0.12,
        -0.02,
        -0.12
    ]
    }
    models = pd.DataFrame(models_dict)
    return models

# =======================================================================================================
# models END
# models TO best_model
# best_model START
# =======================================================================================================

def best_model():
    '''
    Returns a pandas dataframe with the baseline and the best model within the group

    INPUT:
    NONE

    OUTPUT:
    A pandas dataframe with the best model score
    '''
    models_dict = {
    'type' : [
        'baseline',
        'Random Forest Classifier4'
    ],
    'test_accuracy' : [
        0.44,
        0.89,
    ]
    }
    models = pd.DataFrame(models_dict)
    return models

# =======================================================================================================
# best_model END
# best_model TO acquire
# acquire START
# =======================================================================================================

def acquire():
    '''
    Obtains the vanilla version of both the red and white wine dataframe

    INPUT:
    NONE

    OUTPUT:
    red = pandas dataframe with red wine data
    white = pandas dataframe with white wine data
    '''
    red = pd.read_csv('https://query.data.world/s/k6viyg23e4usmgc2joiodhf2pvcvao?dws=00000')
    white = pd.read_csv('https://query.data.world/s/d5jg7efmkn3kq7cmrvvfkx2ww7epq7?dws=00000')
    return red, white

# =======================================================================================================
# acquire END
# acquire TO prepare
# prepare START
# =======================================================================================================

def prepare():
    '''
    Takes in the vanilla red and white wine dataframes and returns a cleaned version that is ready 
    for exploration and further analysis

    INPUT:
    NONE

    OUTPUT:
    wines = pandas dataframe with both red and white wine prepped for exploration
    '''
    red, white = acquire()
    white['wine_color'] = 'white'
    red['wine_color'] = 'red'
    wines = pd.concat([red, white], axis=0)
    return wines

# =======================================================================================================
# prepare END
# prepare TO visual5
# visual5 START
# =======================================================================================================

def visual5():
    '''
    Returns the 5th specific visual for the final_report.ipynb

    INPUT:
    NONE

    OUTPUT:
    scatterplot of the wine's perceived balance vs. wine's quality
    '''
    wines = pd.read_csv('wines.csv', index_col=0)
    original_wines = wines.copy()
    scale_cols = wines.drop(columns=['quality', 'wine_color']).columns.to_list()
    scaler = MinMaxScaler()
    scaler.fit(original_wines[scale_cols])
    original_wines[scale_cols] = scaler.transform(original_wines[scale_cols])
    new_wines = original_wines
    tartness_for_cluster_wines = new_wines[['fixed acidity',
                                            'volatile acidity',
                                            'citric acid',
                                            'pH']]
    structure_for_cluster_wines = new_wines[['total sulfur dioxide',
                                            'pH',
                                            'alcohol']]
    aroma_for_cluster_wines = new_wines[['volatile acidity',
                                        'citric acid',
                                        'sulphates',
                                        'alcohol']]
    taste_for_cluster_wines = new_wines[['fixed acidity',
                                        'volatile acidity',
                                        'residual sugar',
                                        'chlorides',
                                        'pH',
                                        'sulphates',
                                        'alcohol']]
    flavor_for_cluster_wines = new_wines[['fixed acidity',
                                            'volatile acidity',
                                            'citric acid',
                                            'residual sugar',
                                            'chlorides',
                                            'free sulfur dioxide',
                                            'total sulfur dioxide',
                                            'density',
                                            'pH',
                                            'sulphates',
                                            'alcohol']]
    mouthfeel_for_cluster_wines = new_wines[['density',
                                            'pH',
                                            'alcohol']]
    alcohol_for_cluster_wines = new_wines[['residual sugar',
                                            'alcohol']]
    acidity_for_cluster_wines = new_wines[['fixed acidity',
                                            'volatile acidity',
                                            'citric acid',
                                            'pH']]
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(tartness_for_cluster_wines)
    wines['tartness_cluster'] = kmeans.predict(tartness_for_cluster_wines)
    kmeans.fit(structure_for_cluster_wines)
    wines['structure_cluster'] = kmeans.predict(structure_for_cluster_wines)
    kmeans.fit(aroma_for_cluster_wines)
    wines['aroma_cluster'] = kmeans.predict(aroma_for_cluster_wines)
    kmeans.fit(taste_for_cluster_wines)
    wines['taste_cluster'] = kmeans.predict(taste_for_cluster_wines)
    kmeans.fit(flavor_for_cluster_wines)
    wines['flavor_cluster'] = kmeans.predict(flavor_for_cluster_wines)
    kmeans.fit(mouthfeel_for_cluster_wines)
    wines['mouthfeel_cluster'] = kmeans.predict(mouthfeel_for_cluster_wines)
    kmeans.fit(alcohol_for_cluster_wines)
    wines['alcohol_cluster'] = kmeans.predict(alcohol_for_cluster_wines)
    kmeans.fit(acidity_for_cluster_wines)
    wines['acidity_cluster'] = kmeans.predict(acidity_for_cluster_wines)
    wines['agg_score'] = (wines.tartness_cluster +
                      wines.structure_cluster +
                      wines.aroma_cluster +
                      wines.taste_cluster +
                      wines.flavor_cluster +
                      wines.mouthfeel_cluster +
                      wines.alcohol_cluster +
                      wines.acidity_cluster) / 8
    sns.scatterplot(data=wines, x='agg_score', y='quality')
    plt.xlabel('Estimated Wine Balance')
    plt.ylabel('Wine Quality Rating')
    plt.title('Wine Balance vs. Wine Quality')
    plt.show()

# =======================================================================================================
# visual5 END
# =======================================================================================================