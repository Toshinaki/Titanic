#!/usr/bin/env python -W ignore::DeprecationWarning
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
##############################################################################
## Imports and Configs
##############################################################################
# math
import pandas as pd
import numpy as np
from scipy import stats
# typing
from typing import Union, List, Callable, Optional, Tuple
from numbers import Number
# plot
import colorlover as cl
import plotly.offline as plt
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
plt.init_notebook_mode(connected=True)
# datapreparation and model selection
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler, MinMaxScaler, MaxAbsScaler, LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.model_selection import cross_val_predict, StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, log_loss
# classification
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid, RadiusNeighborsClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, PassiveAggressiveClassifier, Perceptron, RidgeClassifier, RidgeClassifierCV, SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
# plotly default colorscales
# ['Blackbody', 'Bluered', 'Blues', 'Earth', 'Electric', 
#  'Greens', 'Greys', 'Hot', 'Jet', 'Picnic', 'Portland', 
#  'Rainbow', 'RdBu', 'Reds', 'Viridis', 'YlGnBu', 'YlOrRd']
#
# colorlover colors
# https://plot.ly/ipython-notebooks/color-scales/

##############################################################################
## Functions and Classes
##############################################################################

###############################################
## helper funcions
def df_contingency_table(df: pd.DataFrame, col1: str, col2: str, ttype: str = 'count') -> pd.DataFrame:
    '''Docstring of `df_contingency_table`

    Construct a contigency table with two columns of given DataFrame.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
    
    Returns:
        crosstab: DataFrame.
    '''
    ct = pd.crosstab(df[col1], df[col2])
    # add sum column and row
    ct['Total'] = ct.sum(axis=1)
    ct.loc['Total'] = ct.sum()
    if ttype == 'count': pass
    elif ttype == 'colper': # row percentage
        ct = (ct / ct.loc['Total'] * 100).round().astype(int)
    return ct

def df_chi_square(df: pd.DataFrame, col1: str, col2: str, cramersv: bool = True) -> str:
    '''Docstring of `df_chi_square`

    Run chi-square test on two columns of given pandas DataFrame,
    with a contigency table calculated on the two columns.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        cramersv: Whether to calculate the Cramér’s V.
            Defaults to True.
    
    Returns:
        A string describes the result of the chi-square test.
    '''
    cto = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, exp = stats.chi2_contingency(cto)
    if cramersv:
        cramersv = '<br>Cramér’s V = {:.2f}'.format(corrected_cramers_V(chi2, cto))
    return 'χ2({dof}) = {chi2:.2f}<br>p = {p:.3f}{cramersv}'.format(dof=dof, chi2=chi2, p=p, cramersv=cramersv or '' )

def df_chi_square_matrix(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    '''Docstring of `df_chi_square_matrix`

    Run chi-square test on every two columns of given pandas DataFrame,
    with contigency tables calculated on the two columns.

    Args:
        df: A pandas DataFrame.
        columns: A list contains columns to run chi-square test with.
    
    Returns:
        A pandas DataFrame of strings descibe the results of chi-square test.
    '''
    dfr = pd.DataFrame(index=columns, columns=columns)
    for i in range(len(columns)):
        for j in range(len(columns)):
            dfr.iloc[i, j] = df_chi_square(df, columns[i], columns[j])
    return dfr

def cramers_V(chi2: float, ct: pd.DataFrame) -> float:
    '''Docstring of `cramers_V`.

    Original algorithm of Cramer's V.

    Args:
        chi2: The chi-square statistic of `ct`.
        ct: The contigency table to calculate Cramer's V for .
    
    Returns:
        The Cramer's V statistic.
    '''
    r, k = ct.shape
    return np.sqrt(chi2 / (ct.values.sum() * (min(r, k) - 1)))

def corrected_cramers_V(chi2: float, ct: pd.DataFrame) -> float:
    '''Docstring of `corrected_cramers_V`.

    Corrected algorithm of Cramer's V.
    See https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V for more details.

    Args:
        chi2: The chi-square statistic of `ct`.
        ct: The contigency table to calculate Cramer's V for .
    
    Returns:
        The Cramer's V statistic.
    '''
    n = ct.values.sum()
    phi2 = chi2/n
    r, k = ct.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min( (kcorr-1), (rcorr-1)))

def grouped_col(df: pd.DataFrame, col1: str, col2: str) -> Tuple[np.array, List[pd.Series]]:
    '''Docstring of `grouped_col`

    Group one column based on the unique values of another.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
    
    Returns:
        A (2, ) tuple contains a numpy array and a list.
        The array contains the unique values.
        The list contains grouped pandas Series.
    '''
    vals = df[col1].unique()
    return vals, [df[df[col1]==val][col2].dropna() for val in vals]

def advanced_describe(data: list, index=None, dtype=None, name: str = None) -> pd.Series:
    '''Docstring of `advanced_describe`

    Descriptive statistics of given data.
    Those are "count", "mean", "std", "min", "25%-50%-75% quantile", "max", "variance", "skewness", "kurtosis".

    Args:
        data: A pandas Series or a list or a numpy array of numerical data.
        index: array-like or Index (1d)
        dtype: numpy.dtype or None
        name: The name of the data.

    Returns:
        The descriptive statistics of given data
    '''
    data = pd.Series(data, index=index, dtype=dtype, name=name)
    des = data.describe()
    t = stats.describe(data)
    des['variance'] = t.variance
    des['skewness'] = t.skewness
    des['kurtosis'] = t.kurtosis
    des = np.r_[[['', name]], [['Item', 'Statistic']], des.reset_index().values]
    return des

###############################################
## plotly functions
def make_colorscale(cname: str, n: int, cnum: str = '3', ctype: str = 'seq') -> list:
    n = n // 10 * 10
    return [[i/(n-1), c] for i, c in  enumerate(cl.to_rgb(cl.interp(cl.scales[cnum][ctype][cname], n)))]

def plotly_df_categorical_bar(df: pd.DataFrame, columns: List[str], ncols: int = 4, save: bool = False, filename: str = 'bar_charts', **kwargs) -> None:
    '''Docstring of `plotly_df_categorical_bar`

    Plot categorical columns of given DataFrame with plotly.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = int(np.ceil(len(columns) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=columns, print_grid=False)
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = df[columns[ncols * i + j]].value_counts()
            except:
                break
            trace = go.Bar(x=s.index, y=s.values, name=s.name)
            fig.append_trace(trace, i+1, j+1)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)

def plotly_df_numerical_hist(df: pd.DataFrame, columns: List[str], ncols: int = 4, save: bool = False, filename: str = 'histograms', **kwargs) -> None:
    '''Docstring of `plotly_df_numerical_hist`

    Plot numerical columns of given DataFrame with plotly.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = int(np.ceil(len(columns) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=columns, print_grid=False)
    
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = df[columns[ncols * i + j]]
            except:
                break
            trace = go.Histogram(x=s, xbins=dict(start=s.min(), end=s.max(), size=kwargs.get('size', None)), name=s.name)
            fig.append_trace(trace, i+1, j+1)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)

def plotly_df_grouped_hist(df: pd.DataFrame, col1: str, col2: str, ncols: int = 4, normdist: bool = False, save: bool = False, filename: str = 'histograms') -> None:
    '''Docstring of `plotly_df_grouped_hist`

    Plot histograms of given column grouped on the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
        ncols: Number of subplots of every row.
        normdist: Whether plot the normal distribution with mean 
            and std of given data.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    vals, data = grouped_col(df, col1, col2)
    nrows = int(np.ceil(len(vals) / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=[str(v) for v in vals],
                            shared_yaxes=True, print_grid=False)
    layout = go.Layout({'title': 'Histogram of {} - grouped by {}'.format(col2, col1), 'showlegend': False})
    for k in range(nrows):
        layout['yaxis{}'.format(k+1)]['title'] = 'Count'
    
    for i in range(nrows):
        for j in range(ncols):
            try:
                s = data[ncols * i + j]
            except:
                break
            if normdist:
                tempfig = ff.create_distplot([s], [s.name], curve_type='normal', histnorm='')
                for trace in tempfig['data']:
                    trace['xaxis'] = 'x{}'.format(i+1)
                    trace['yaxis'] = 'y{}'.format(j+1)
                    if trace['type'] == 'scatter':
                        trace['y'] = trace['y'] * s.count()
                    fig.append_trace(trace, i+1, j+1)
            else:
                trace = go.Histogram(x=s, xbins=dict(start=s.min(), end=s.max(), size=kwargs.get('size', None)), name=v)
                fig.append_trace(trace, i+1, j+1)
    fig['layout'].update(layout)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)
    
#######################
def plotly_df_crosstab_heatmap(df: pd.DataFrame, col1: str, col2: str, ttype: str = 'count', title: bool = False, axes_title: bool = False, save: bool = False, filename: str = 'crosstab_heatmap') -> None:
    '''Docstring of `plotly_df_crosstab_heatmap`

    Plot contigency table of two given columns with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        title: Whether to show the plot title or not.
        axes_title: Whether to show the axis' title or not.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ct = df_contingency_table(df, col1, col2, ttype=ttype)
    fig = ff.create_annotated_heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index))
    fig['layout']['title'] = title and '{}-{}'.format(ct.index.name, ct.columns.name)
    fig['layout']['xaxis']['title'] = axes_title and ct.columns.name
    fig['layout']['yaxis']['title'] = axes_title and ct.index.name
    fig['layout']['xaxis']['side'] = 'bottom'
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)

def plotly_df_crosstab_heatmap_matrix(df: pd.DataFrame, columns: List[str], ttype: str = 'count', colorscale: Union[str, list] = 'Greens', width: int = 950, height: int = 750, save: bool = False, filename: str = 'crosstab_heatmap_matrix') -> None:
    '''Docstring of `plotly_df_crosstab_heatmap_matrix`

    Plot contigency tables of every two given columns with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        ttype: Determines how the contigency table is calculated.
            'count': The counts of every combination.
            'colper': The percentage of every combination to the 
            sum of every rows.
            Defaults to 'count'.
        colorscale: The color scale to use.
        width: Plot width.
        height: Plot height.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = ncols = len(columns)
    fig = tls.make_subplots(rows=nrows, cols=ncols, 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    layout = go.Layout({'title': 'Contingency Table Matrix', 'annotations': [], 'width': width, 'height': height})
    for k in range(nrows):
        layout['xaxis{}'.format(k+1)]['title'] = columns[k]
        layout['yaxis{}'.format(k+1)]['title'] = columns[k]
        layout['xaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
        
    for i in range(nrows):
        for j in range(ncols):
            ct = df_contingency_table(df, columns[i], columns[j], ttype=ttype)
            
            annheat = ff.create_annotated_heatmap(z=ct.values, x=list(ct.columns), y=list(ct.index))
            trace = annheat['data'][0]
            trace['colorscale'] = colorscale

            annotations = annheat['layout']['annotations']
            for ann in annotations:
                ann['xref'] = 'x{}'.format(j+1)
                ann['yref'] = 'y{}'.format(i+1)
                ann['font']['color'] = float(ann['text']) / df.shape[0] > 0.5 and 'rgb(255,255,255)' or 'rgb(0,0,0)'
                if ttype == 'colper': ann['text'] = ann['text'] + '%'
            layout['annotations'].extend(annotations)
            
            fig.append_trace(trace, i+1, j+1)    
            
    fig['layout'].update(layout)
    save and plt.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plt.iplot(fig)

def plotly_df_crosstab_stacked(df: pd.DataFrame, col1: str, col2: str, save: bool = False, filename: str = 'crosstab_stacked_bar') -> None:
    '''Docstring of `plotly_df_crosstab_stacked`

    Plot stacked bar of two given columns' contigency table with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        col1: Index of the contigency table.
        col2: Column of the contigency table.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ct = df_contingency_table(df, col1, col2)
    layout = go.Layout(
        barmode = 'stack',
        title = '{}-{}'.format(ct.index.name, ct.columns.name),
        yaxis = dict(title=ct.columns.name),
        annotations = [
            dict(
                x=1.12,
                y=1.05,
                text='Pclass',
                showarrow=False,
                xref="paper",
                yref="paper",
            )
        ]
    )
    ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
    ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
    ct = (ct / ct.iloc[-1] * 100).round().astype(int)
    data = [go.Bar(x=ct.iloc[i][:-1], y=ct.columns[:-1], name=ct.index[i], orientation='h') for i in range(ct.index.shape[0]-1)]
    
    fig = go.Figure(data=data, layout=layout)
    save and plt.plot(fig, filename=filename+'html', auto_open=False)
    plt.iplot(fig)

def plotly_df_crosstab_stacked_matrix(df: pd.DataFrame, columns: List[str], colorscale: Union[str, list] = 'Greens', width: int = 950, height: int = 750, save: bool = False, filename: str = 'crosstab_stacked_matrix') -> None:
    '''Docstring of `plotly_df_crosstab_stacked_matrix`

    Plot stacked bars of every two given columns' contigency table with plotly heatmap.

    Args:
        df: A pandas DataFrame.
        columns: The column names.
        colorscale: The color scale to use.
        width: Plot width.
        height: Plot height.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    nrows = ncols = len(columns)
    fig = tls.make_subplots(rows=nrows, cols=ncols, 
                            shared_xaxes=True, shared_yaxes=True, 
                            vertical_spacing=0.01, horizontal_spacing=0.01, print_grid=False)
    layout = go.Layout({'title': 'Stacked Bar Matrix', 'annotations': [], 
                        'width': width, 'height': height, 'barmode': 'stack',
                        'showlegend': False, 'hoverlabel': {'bgcolor': 'black', 'font': {'color': 'white'}, 'namelength': -1}})
    for k in range(nrows):
        layout['xaxis{}'.format(k+1)]['title'] = columns[k]
        layout['yaxis{}'.format(k+1)]['title'] = columns[k]
        #layout['xaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['type'] = 'category'
        layout['yaxis{}'.format(k+1)]['autorange'] = 'reversed'
    for i in range(nrows):
        for j in range(ncols):
            ct = df_contingency_table(df, columns[j], columns[i])
            ct.index = ct.index.astype(str) + ' <br>(n=' + ct['Total'].astype(str) + ')'
            ct.columns = ct.columns.astype(str) + ' <br>(n=' + ct.iloc[-1].astype(str) + ')'
            ct = (ct / ct.iloc[-1] * 100).round().astype(int)
            data = [go.Bar(x=ct.iloc[k][:-1], y=ct.columns[:-1], name=ct.index[k], orientation='h') for k in range(ct.index.shape[0]-1)]
            
            for trace in data:
                fig.append_trace(trace, i+1, j+1)
    
    fig['layout'].update(layout)
    save and plt.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plt.iplot(fig)
    
#######################
def plotly_df_box(df: pd.DataFrame, col1: str, col2: str, save: bool = False, filename: str = 'Box Plot') -> None:
    '''Docstring of `plotly_df_crosstab_stacked`

    Plot box-plot of one column grouped by the unique value of another column with plotly.

    Args:
        df: A pandas DataFrame.
        col1: Name of the column that the unique values of which
            will be used for grouping.
        col2: Name of the column that will be grouped.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    # df = df[[col1, col2]].dropna()
    # cols = df[col1].unique()
    # traces = [go.Box(y=df[df[col1] == col][col2], boxmean='sd', name=col) for col in cols]
    vals, data = grouped_col(df, col1, col2)
    traces = [go.Box(y=d, boxmean='sd', name=v) for v,d in zip(vals, data)]

    layout = go.Layout(
        title='{} boxes grouped by {}'.format(col2, col1),
        yaxis=dict(title=col2),
        xaxis=dict(title=col1)
    )
    fig = go.Figure(data=traces, layout=layout)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)
    
def plotly_df_chi_square_matrix(df: pd.DataFrame, columns: List[str], save: bool = False, filename: str = 'Chi-Square Matrix') -> None:
    '''Docstring of `plotly_df_chi_square_matrix`

    Run chi-square test on every two columns of given pandas DataFrame,
    with contigency tables calculated on the two columns.
    Then plot the results as a matrix.

    Args:
        df: A pandas DataFrame.
        columns: A list contains columns to run chi-square test with.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    data = np.c_[columns, df_chi_square_matrix(df, columns).values]
    data = np.r_[[['']+columns], data]
    fig = ff.create_table(data, height_constant=45, index=True)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)

def plotly_describes(data: list, names: list = [], save: bool = False, filename: str = 'Descriptive Statistics'):
    '''Docstring of `plotly_describes`

    Plot a table of descriptive statistics of given data with plotly.

    Args:
        data: A list of numerical data.
        names: A list contains names corresponding to data.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ndata = len(data)
    names = names or ['']*ndata
    describes = np.empty((12, ndata+1), dtype=object)
    for i, d, n in zip(range(ndata), data, names):
        des = advanced_describe(d, name=n)
        if i == 0:
            describes[0, 0] = 'Describes'
            describes[1:, 0] = des[2:, 0]
        describes[0, i+1] = des[0, 1]
        describes[1:, i+1] = [int(v*10000)/10000 for v in des[2:, 1]]
    fig = ff.create_table(describes, index=True)
    save and plt.plot(fig, filename=filename+'.html', auto_open=False)
    plt.iplot(fig)

def plotly_qq_plots(data: list, names: list = [], ncols: int = 4, save: bool = False, filename: str = 'QQ plots'):
    '''Docstring of `plotly_describes`

    Plot QQ-plots of given data with plotly.

    Args:
        data: A list of numerical data.
        names: A list contains names corresponding to data.
        ncols: Number of subplots of every row.
        save: Whether to save the plot or not.
        filename: Save the plot with this name.
    '''
    ndata = len(data)
    names = names or ['']*ndata
    nrows = int(np.ceil(ndata / ncols))
    fig = tls.make_subplots(rows=nrows, cols=ncols, subplot_titles=names, 
                            vertical_spacing=0.1, horizontal_spacing=0.1, print_grid=False)
    layout = go.Layout({'title': 'QQ plots', 'showlegend': False})
    for i in range(nrows):
        for j in range(ncols):
            p = stats.probplot(data[ncols * i + j])
            fig.append_trace(go.Scatter(x=p[0][0], y=p[0][1], mode='markers'), i+1, j+1)
            fig.append_trace(go.Scatter(x=p[0][0], y=p[0][0]*p[1][0]+p[1][1]), i+1, j+1)
    
    fig['layout'].update(layout)
    save and plt.plot(fig, filename=filename+'.html', image_width=width, image_height=height, auto_open=False)
    plt.iplot(fig)

# Data cleaning:
#   Fix or remove outliers (optional).
#   Fill in missing values (e.g., with zero, mean, median…) or drop their rows (or columns).
# Feature selection (optional):
#   Drop the attributes that provide no useful information for the task.
# Feature engineering, where appropriate:
#   Discretize continuous features.
#   Decompose features (e.g., categorical, date/time, etc.).
#   Add promising transformations of features (e.g., log(x), sqrt(x), x^- [ ] etc.).
#   Aggregate features into promising new features.
# Feature scaling: standardize or normalize features.
##  0 mean and unit variance - scale, StandardScaler
##  0, 1 range - minmax_scale, MinMaxScaler 
##  -1, 1 - maxabs_scale, MaxAbsScaler (for data that is already centered at zero)
##  sparse data - maxabs_scale, MaxAbsScaler; scale, StandardScaler with "with_mean=False"
##  with outliers - robust_scale, RobustScaler


###############################################
## Data Preparation Classes

class DataFrameFiller(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameFiller`.

    Deal with missing values for any 2D pandas DataFrame.

    Args:
        clean_mode: Determines how the missing data being processed.
            Avaiable values are None, "fill", "remove" and "both". Defaults to None.
            If None, default fill_mode will be applied to all columns:
            Integer - "median"; float - "mean"; string - "mode".
            If "remove", rows with missing data at all columns will be removed.
        fill_mode: Required when `clean_mode` set to "fill" or "both".
            Can be passed a single value or a dict.
            The dict must be structured as column_name -> mode.
            Valid modes for all column types are:
            Integer - "median", "mode", any integer or function that returns an integer;
            Float - "mean", "median", "mode", any float or function that returns a float;
            String - "mode", any string or function that returns a string;
            Missmatching modes are replaced by defaults.
            Functions must accept iterable and return values with types specified above.
            When passed a single mode, it will be applied to all columns.
            When passed a `dict`, the fill modes will be applied to the corresponding 
            columns. For the remain columns, if `clean_mode` is "fill", apply defaults,
            else if "remove", apply remove.
            When `case_mode` == "remove", this parameter will be ignored.
    '''
    
    def __init__(self, clean_mode: Optional[str] = None, fill_mode: Union[None, dict, str, int, float, Callable] = None):
        
        assert clean_mode in [None, 'fill', 'remove', 'both'], 'Invalid value for `clean_mode`. Avaliable values are None, "fill", "remove" and "both".'
        if clean_mode in ['both']:
            assert isinstance(fill_mode, dict), 'Invalid parameter. `fill_mode` must be a `dict` when `clean_mode` set to "both".'
        
        self.clean_mode = clean_mode
        self.fill_mode = clean_mode == 'remove' and None or fill_mode
        
    def fit(self, X: Union[pd.DataFrame, np.array], y=None):
        '''Initialize operators.
        
        patterns:
            1. clean_mode = None, use only fill_mode
            2. clean_mode = 'fill', fill_mode is single value, make and check filler dict with all columns;
            fill_mode is dict, check filler and add remain columns
            3. clean_mode = 'remove', ignore fill_mode
            4. clean_mode = 'both', fill_mode must be a dict, check filler and apply remove to remain columns
        
        Args:
            X: Accept a 2D pandas DataFrame, or a 1D Series or numpy array.
        '''
        if self.clean_mode == 'remove':
            self.fillers = 'remove'
        else:
            # make sure self.fill_mode is a dict 
            if not isinstance(self.fill_mode, dict):
                self.fill_mode = {col: self.fill_mode for col in X}
            # check for the validity of fill modes; default value will be used if not valid
            # then calculate the filler values
            fillers = {}
            for col, m in self.fill_mode.items():
                if m in ['mean', 'median', 'mode', None]:
                    valid_modes = self._valid_modes(X[col].dtype)
                    if not m in valid_modes: m = valid_modes[0]
                    m = m == 'mean' and X[col].mean() or (m == 'median' and X[col].median() or X[col].value_counts().index[0])
                if callable(m): m = m(X[col])
                fillers[col] = m
            if self.clean_mode == None:
                self.fillers = fillers
            elif self.clean_mode == 'fill':
                # default fillers for columns
                self.fillers = {col: X[col].dtype == np.dtype('O') and X[col].value_counts().index[0] or ( \
                                     np.issubdtype(X[col].dtype, np.integer) and X[col].median() or X[col].mean()) for col in X}
                self.fillers.update(fillers)
            else: # clean_mode = 'both'; apply remove to remain columns
                self.fillers = [fillers]
        return self
    def transform(self, X, y=None):
        X = X.copy()
        # # drop rows and columns where all values are NA
        # X = X.dropna(axis=0, how='all').dropna(axis=1, how='all')
        # clean missing value
        if self.fillers == 'remove':
            X = X.dropna()
        elif isinstance(self.fillers, list):
            X = X.fillna(value=self.fillers[0])
            X = X.dropna()
        else:
            X = X.fillna(value=self.fillers)
        return X
    
    def _valid_modes(self, dtype):
        return dtype == np.dtype('O') and ['mode',] \
            or (np.issubdtype(dtype, np.integer) and ['median', 'mode'] \
            or ['mean', 'median', 'mode'])

class DataFrameFeatureAdder(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameFeatureAdder`

    Add extra features to DataFrame with given columns and functions.

    Args:
        adds: Tuples with same structure that contains 
            any number of columns' names, a new column name, 
            and a function to generate a new column 
            with given DataFrame and columns.
    '''

    def __init__(self, adds: list, remove: bool = False):
        self.adds = adds
        assert self.adds != [], 'Invalid input. At least one tuple for generating a new column is needed.'
        self.remove = remove

    def fit(self, X: pd.DataFrame, y=None):
        remove = self.remove and []
        for adds in self.adds:
            assert len(adds) == 3, 'Invalid parameters. Exactly 3 parameters are needed.'
            cols, name, func = adds
            for col in cols:
                assert col in X, '"{}" is not a column of given DataFrame'.format(col)
                if self.remove: remove.append(col)
            assert not name in X, '"{}" already exists.'.format(name)
        self.remove = remove
        return self

    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for adds in self.adds:
            cols, name, func = adds
            X[name] = func(X, *cols)
        if self.remove:
            X = X.drop(columns=[*self.remove])
        return X

class DataFrameEncoder(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameEncoder`

    Encodes specified columns with given methods.
    categorical -> numerical / one-hot
    numerical -> range / range one-hot / binarization

    Args:
        cols: Columns' names to encode.
        encoders: Encoders's names for encoding columns 
        corresponding to `cols`.
        newcolnames: A list of lists of string, that are
            encoded columns's names. Optional.
            If None, the combination of original column names 
            and unique values will be used.
            But if any one is specified, the others must be 
            specified with blank lists.
        params: A list of dict containing parameters for 
            each encoder. Optional.
            But if any one is specified, the others must be 
            specified with blank dicts.
        inplace: Whether delete the original columns or not
    '''

    def __init__(self, cols: list, encoders: list, newcolnames: Optional[List[List[str]]] = None, params: Optional[List[dict]] = None, inplace: bool = True):
        assert len(cols) == len(encoders), 'Parameter `cols` and `encoders` must have same length.'
        if newcolnames: assert len(newcolnames) == len(cols), 'Not enough new names. {} is needed.'.format(len(cols))
        else: newcolnames = [[]] * len(cols)
        if params: assert len(params) == len(cols), 'Not enough parameters. {} is needed.'.format(len(cols))
        else: params = [{}] * len(cols)
        self.cols = cols
        self.encoders = encoders
        self.newcolnames = newcolnames
        self.params = params
        self.inplace = inplace
    
    def fit(self, X: pd.DataFrame, y=None):
        encoders = {}
        for col, encoder, newnames, param in zip(self.cols, self.encoders, self.newcolnames, self.params):
            assert col in X, '"{}" is not a column of given DataFrame'.format(col)

            if encoder == 'label': # 1 col to 1 col
                encoder = LabelEncoder().fit(X[col])
                newnames = newnames or [col+'_labeled',]
            elif encoder == 'label2': # 1 col to multi cols
                encoder = LabelBinarizer(**{
                    k: param.get(k, d) for k,d in [
                        ('neg_label', 0),
                        ('pos_label', 1),
                        ('sparse_output', False)
                    ]
                }).fit(X[col])
                newnames = newnames or [col+'_'+ str(c) for c in encoder.classes_]
            elif encoder == '1hot': # 1 col to multi cols
                encoder = OneHotEncoder(**{
                    k: param.get(k, d) for k,d in [
                        ('n_values', 'auto'), ('categorical_features', 'all'), 
                        ('dtype', np.int), ('sparse', False), 
                        ('handle_unknown', 'ignore')]
                }).fit(X[col].values.reshape(-1,1))
                newnames = newnames or [col+'_'+ str(c) for c in sorted(X[col].unique())]
            elif callable(encoder):
                pass
            else:
                raise ValueError('"{}" is not a valid encoder. See help for more information.'.format(encoder))
            encoders[col] = (encoder, newnames)
        self.encoders = encoders
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        for col in self.encoders:
            encoder, newnames = self.encoders[col]
            try:
                newcols = encoder.transform(X[col])
            except:
                newcols = encoder.transform(X[col].values.reshape(-1,1))
            X = X.reindex(columns=X.columns.tolist()+newnames)
            X[newnames] = newcols
        if self.inplace:
            X = X.drop(columns=list(self.encoders.keys()))
        return X

class DataFrameScaler(BaseEstimator, TransformerMixin):
    '''Docstring of `DataFrameScaler`

    Scaling a pandas DataFrame.

    Args:
        scale: Valid values are 
        "unit" - Centers to the mean and component wise scale 
        to unit variance;
        "0,1" - Scales data to given range. Use extra parameter 
        `feature_range=(min,max)` to set range;
        "-1,1" - Scales data to the range [-1, 1] by dividing 
        through the largest maximum value in each feature. 
        It is meant for data that is already centered at zero or 
        sparse data.
        Extra paramters will be passed to sklearn scalers if specified.
        ignore_cols: Columns that will not be scaled.
        By default, all categorical columns will be ignored.
        Specify this parameter to ignore numerical columns too.
    '''

    SCALERS = {
        'unit': StandardScaler,
        '0,1': MinMaxScaler,
        '-1,1': MaxAbsScaler
    }

    def __init__(self, scaler: str = 'unit', ignore_cols: List[str] = [], target_cols: List[str] = [], **kwargs):
        assert scaler in ['unit', '0,1', '-1,1'], 'Invalid scaler {}. See help for valid scalers.'.format(scaler)
        self.scaler = scaler
        self.ignore_cols = ignore_cols
        self.target_cols = ignore_cols and [] or target_cols
        self.kwargs = kwargs
        self.ignore_cols = ignore_cols
        self.target_cols = ignore_cols and [] or target_cols
    
    def fit(self, X: pd.DataFrame, y=None):
        if self.scaler == 'unit':
            self.scaler = StandardScaler(**{
                k: self.kwargs.get(k, d) for k,d in [('copy', True), ('with_mean', True), ('with_std', True)]
            })
        elif self.scaler == '0,1':
            self.scaler = MinMaxScaler(**{
                k: self.kwargs.get(k, d) for k, d in [('feature_range', (0, 1)), ('copy', True)]
            })
        elif self.scaler == '-1,1':
            self.scaler = MaxAbsScaler(**{
                k: self.kwargs.get(k, d) for k, d in [('copy', True)]
            })
        # self.scaler = self.SCALERS[self.scaler](**self.kwargs).fit(X)
        self.target_cols = self.ignore_cols and [col for col in X.select_dtypes(include=['number']) if not col in self.ignore_cols] or (
            self.target_cols and [col for col in X.select_dtypes(include=['number']) if col in self.target_cols] or X.select_dtypes(include=['number']).columns)
        self.scaler.fit(X[self.target_cols])
        return self
    
    def transform(self, X: pd.DataFrame, y=None):
        X = X.copy()
        X[self.target_cols] = self.scaler.transform(X[self.target_cols])
        return X


###############################################
## Model Selection
def select_classifier(X, y, n_splits=10, test_size=0.1, random_state=0, show=True):
    classifiers = [
        AdaBoostClassifier(),
        BaggingClassifier(),
        BernoulliNB(),
        CalibratedClassifierCV(),
        DecisionTreeClassifier(),
        ExtraTreeClassifier(),
        GaussianNB(),
        GaussianProcessClassifier(),
        GradientBoostingClassifier(),
        KNeighborsClassifier(),
        LinearDiscriminantAnalysis(),
        LinearSVC(),
        LogisticRegression(),
        LogisticRegressionCV(),
        MLPClassifier(),
        MultinomialNB(),
        NearestCentroid(),
        NuSVC(),
        PassiveAggressiveClassifier(),
        Perceptron(),
        QuadraticDiscriminantAnalysis(),
        RadiusNeighborsClassifier(),
        RandomForestClassifier(),
        RidgeClassifier(),
        RidgeClassifierCV(),
        SGDClassifier(),
        SVC()
    ]
    names = [clf.__class__.__name__ for clf in classifiers]
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
    scores = {}
    for train_index, test_index in cv.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        for name, clf in zip(names, classifiers):
            try:
                clf.fit(X_train, y_train)
                train_predictions = clf.predict(X_test)
                acc = accuracy_score(y_test, train_predictions)
            except:
                acc = 0
            s = scores.get(name, [])
            s.append(acc)
            scores[name] = s
    scores = [[n, np.mean(s)] for n, s in scores.items()]
    scores = pd.DataFrame(scores, columns=['Classifier', 'Score']).sort_values(by='Score', ascending=False)
    if show:
        print(scores)
    return scores.iloc[0, 0], classifiers[scores.iloc[0].name], scores