import numpy as np
import scipy
import scipy.io
import scipy.linalg

import seaborn as sns

import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import pandas as pd

import torch

import os

def create_synthData(N=400, outDir='./', device='cpu'):
    '''
    creating Main paper Synth data,
    N : number of data
    F$ : number of features in view $ 
    '''
    views  = []
    k  = 2   # Number of latent features
    F1 = 2   
    F2 = 2   
    F3 = 2 

    # First half of points belong to class 1, second to class 2
    G = np.zeros( ( N, k ) )
    
    G[:int(N/2),0] = 1.0
    G[int(N/2):,1] = 1.0
    classes = ['Class1' for i in range(int(N/2))] + ['Class2' for i in range(int(N/2))] 

    # Each class lies on a different concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (N, 1) )
    rand_noise = 0.1 * np.random.randn(N, k)
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius     = G.dot(np.asarray( [[1.0], [2.0]] )).reshape( (N, 1) )
    V1    = np.hstack([radius, radius]) * circle_pos + rand_noise
    views.append(V1)
    
    # Each class lies on a different parabola
    rand_x     = np.random.uniform(-3.0, 3.0, (N, 1) )
    rand_noise = 0.1 * np.random.randn(N, k)
    intercepts = G.dot( np.asarray([[0.0], [1.0]])).reshape( (N, 1) )
    quadTerms  = G.dot( np.asarray( [[2.0], [0.5]] )).reshape( (N, 1) )
    V2    = np.hstack( [rand_x, intercepts + quadTerms * (rand_x*rand_x)]) + rand_noise
    views.append(V2) 

    # Class on inside is drawn from a gaussian, class on outside is on a concentric circle
    rand_angle = np.random.uniform(0.0, 2.0 * math.pi, (N, 1) )
    inner_rand_noise = 1.0 * np.random.randn(int(N/2), k) # More variance on inside
    outer_rand_noise = 0.1 * np.random.randn(int(N/2), k)
    rand_noise = np.vstack( [outer_rand_noise, inner_rand_noise] )
    circle_pos = np.hstack( [np.cos(rand_angle), np.sin(rand_angle)])
    radius     = G.dot(np.asarray( [[2.0], [0.0]] )).reshape( (N, 1) )
    V3    = np.hstack([radius, radius]) * circle_pos + rand_noise
    views.append(V3)

    # We have no missing data
    K = np.ones( (N, 3) )
    
    # Gather into dataframes to plot

    dfs = []
    for v in views:
      df = pd.DataFrame(v, columns=['x', 'y'])
      df['Classes'] = classes
      dfs.append(df)
    
    # Plot to PDF
    with PdfPages(os.path.join(outDir, 'originalData.pdf')) as pdf:
      for viewIdx, df in enumerate(dfs):
        fig = sns.lmplot(x="x", y="y", fit_reg=False, markers=['+', 'o'], legend=False, hue="Classes", data=df).fig
        plt.legend(loc='best')
        plt.title('View %d' % (viewIdx))
        pdf.savefig()
        plt.close(fig)
  
    views = [torch.tensor(view).to(device) for view in views]
    return views

if __name__ == '__main__':
   create_synthData() 