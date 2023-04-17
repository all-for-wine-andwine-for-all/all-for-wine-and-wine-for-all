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