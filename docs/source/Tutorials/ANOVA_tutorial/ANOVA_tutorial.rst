ANOVA 
-----

.. code:: python

    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    import Analysis.Anova as anova
    
    
    %matplotlib inline 
    %load_ext autoreload
    %autoreload 2

.. code:: python

    
    groups = []
    for mean in [1, 1.5, 1.5, 1]:
        groups.append(np.random.normal(mean, 0.5, 100))
    
    groups = pd.DataFrame(groups).T
    ax = sns.boxplot(groups)



.. image:: output_1_1.png


.. code:: python

    anova.one_way(groups)

.. parsed-literal::

    +-----------+-------------+--------------+-------------+-------------+------------+
    |   F-value |     p-value |   effect sss |   effect df |   error sss |   error df |
    +===========+=============+==============+=============+=============+============+
    |   35.4803 | 1.11022e-16 |      24.6688 |           3 |     91.7773 |        396 |
    +-----------+-------------+--------------+-------------+-------------+------------+





.. parsed-literal::

    (27.174986167650943, 5.5511151231257827e-16, 3, 396)


.. code:: python

    #Conduct anova on one population which was exposed to different levels of two treatments
    
    factorByfactor = np.array([[np.random.normal(1, 0.5, 100), np.random.normal(1.5, 0.5, 100), np.random.normal(1, 0.5, 100)],
                              [np.random.normal(1, 0.5, 100), np.random.normal(1.5, 0.5, 100), np.random.normal(1, 0.5, 100)]]
                             ).swapaxes(0,2).swapaxes(1,2)
    
    anova.two_way(factorByfactor, 'seed', 'fertilizer')



.. parsed-literal::

    +-------------+---------------+------+------------+-------------+
    | Source      |   Mean square |   df |   F-values |    p-values |
    +=============+===============+======+============+=============+
    | seed        |      0.750668 |    1 |    3.24798 | 0.0720186   |
    +-------------+---------------+------+------------+-------------+
    | fertilizer  |     20.3989   |    2 |   88.2614  | 1.11022e-16 |
    +-------------+---------------+------+------------+-------------+
    | Interaction |      0.234242 |    2 |    1.01351 | 0.363568    |
    +-------------+---------------+------+------------+-------------+



