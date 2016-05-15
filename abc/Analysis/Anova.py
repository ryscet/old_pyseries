"""
Anova
=====

Statistical tools for time-series analysis.

    * One-way: Find time intervals where signals recorded under single conditions differ from the baseline.
    * Two-way: Find interactions between varying conditions time intervals of the recorded signal.
    * Repeated-measures: Find time intervals where the signal was systematically changing on a group level.

"""


import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt
from tabulate import tabulate
import pandas as pd
import seaborn as sns


def one_way(groups):
    """Run one way analysis of variance on n groups of equal length.
       * Identify which groups significanlty deviate from  the grand mean.
       * Prints a table with a spss-style output.


    Parameters
    ----------
    group: list or ndarray
        | If list then each index represents a group,
        | If ndarray then each column represents a group.

    Returns
    -------
    F: double
        F-value, ratio between effect and error sum of squares.
    p: double
        Probability of obtaining F-value by chance.
    df_effect: int
        degrees of freedom for the effect (n groups -1).
    df_error: int
        degrees of freedom for the error (n groups * (n samples - 1)).
    """


    groups = np.array(groups).T


    n_samples = groups.shape[0]
    n_groups = groups.shape[1]


    #total_sumsq = np.sum([(x- groups.ravel().mean())**2 for x in groups.ravel()])

    within_group_sumsq = np.sum([[(x - group.mean())**2] for group in groups.T for x in group])

    between_group_sumsq = np.sum([ n_samples * ((group.mean()- groups.mean())**2) for group in groups.T])

    df_within = n_groups * (n_samples-1)
    df_between = n_groups-1

    F = (between_group_sumsq / df_between) / (within_group_sumsq / df_within )

    p = 1 - f.cdf(F, df_between,df_within)


    sns.boxplot(pd.DataFrame(groups))

    print(tabulate([[F, p, between_group_sumsq, df_between,  within_group_sumsq, df_within]],
                     ['F-value','p-value','effect sss','effect df','error sss', 'error df'], tablefmt="grid"))


    return F, p, df_between, df_within


def plot_F_probability(dfn, dfd, F):
    x = np.linspace(0, F + 1, 1001)[1:]


    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(x, 1-f.cdf(x, dfn, dfd),  '--', label=r'$df_1=%i,\ df_2=%i$' % (dfn, dfd))
    ax.set_ylabel(r'$ 1 - cdf(df_1,df_2)$')
    ax.set_xlabel('$x (F = %f)$' %F)
    ax.set_title('F-distribution')
    print(1-f.cdf(F, dfn, dfd))

    plt.legend()
    plt.show()

tmp = np.array([[[4,6,8], [6,6,9], [8,9,13]],
              [[4,8,9], [7,10,13], [12,14,16]]]).swapaxes(0,2).swapaxes(1,2)

def two_way(data, f1_name, f2_name):

    """Run two way analysis of variance in a factor by factor design.
       * Identify main effects for each factor.
       * Identify interaction between factors.
       * Print a table with a spss-style output.


    Parameters
    ----------
    data: ndarray
        | Each row represents a 1st factor level.
        | Each column respresents a 2nd factor level.
        | Each layer (depth dimension) is an observation.

    """


    #Sums of squares
    factor_1_effect, factor_2_effect, within_error  = factor_sumofsq(data)

    total_sumofsq = np.sum((data.ravel() - data.mean())**2)

    interaction_sumofsq = total_sumofsq - factor_1_effect - factor_2_effect - within_error

    #degrees of freedom
    factor_1_df, factor_2_df  = data.shape[1]-1, data.shape[2]-1
    error_df = (data.shape[0]-1) * (data.shape[1] * data.shape[2])
    interaction_df = factor_1_df * factor_2_df
    #total_df = factor_1_df + factor_2_df + error_df + interaction_df

    #Mean squares
    within_mean_ssq = within_error / error_df
    f1_mean_ssq, f2_mean_ssq = factor_1_effect / factor_1_df, factor_2_effect / factor_2_df
    interaction_ssq = interaction_sumofsq / interaction_df

    #F values
    F1, F2 = f1_mean_ssq / within_mean_ssq, f2_mean_ssq / within_mean_ssq
    F_interaction = interaction_ssq / within_mean_ssq

    #P values
    p_F1 =  1 - f.cdf(F1, factor_1_df, error_df)
    p_F2 =  1 - f.cdf(F2, factor_2_df, error_df)
    p_interaction =  1 - f.cdf(F_interaction, interaction_df, error_df)

    print (tabulate([[f1_name, f1_mean_ssq, factor_1_df, F1, p_F1],
                     [f2_name, f2_mean_ssq,factor_2_df, F2, p_F2],
                     ['Interaction', interaction_ssq, interaction_df, F_interaction, p_interaction]],
                     ['Source','Mean square','df','F-values', 'p-values'], tablefmt='grid'))



    #return [F1, p_F1, F2, p_F2, F_interaction, p_interaction]


def factor_sumofsq(data):

    f1_effect_sumofsq = 0
    f2_effect_sumofsq = 0
    error_sumofsq = 0

    #iterate over levels of the 1st factor
    for factor1_level in data.swapaxes(0,1):

        f1_effect_sumofsq = f1_effect_sumofsq  + ((factor1_level.mean() - data.mean())**2) * len(factor1_level.ravel())
        error_sumofsq = error_sumofsq + np.sum([[(x - other_factor.mean())**2] for other_factor in factor1_level.T for x in other_factor])

    #iterate over levels of the 2nd factor
    for factor2_level in data.swapaxes(1,2).swapaxes(0,1):

        f2_effect_sumofsq = f2_effect_sumofsq  + ((factor2_level.mean() - data.mean())**2) * len(factor2_level.ravel())





    return f1_effect_sumofsq,f2_effect_sumofsq, error_sumofsq

