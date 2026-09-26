"""Twenty-four named regression methods; configurations are not counted as methods."""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, SplineTransformer
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge,Lasso,ElasticNet,BayesianRidge,ARDRegression,HuberRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,BaggingRegressor,GradientBoostingRegressor,HistGradientBoostingRegressor,AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

class StableOLS(RegressorMixin,BaseEstimator):
    """Minimum-norm least squares, explicit rank tolerance for redundant dummy columns."""
    def fit(self,X,y):
        self.coef_=np.linalg.lstsq(np.c_[np.ones(len(X)),X],y,rcond=1e-10)[0]
        return self
    def predict(self,X):return np.c_[np.ones(len(X)),X]@self.coef_

METHODS={
 'OLS':('Linear',[{}]),'Ridge':('Linear',[{'alpha':1},{'alpha':10}]),
 'Lasso':('Linear',[{'alpha':.001},{'alpha':.01}]),
 'ElasticNet':('Linear',[{'alpha':.001,'l1_ratio':.5},{'alpha':.01,'l1_ratio':.5}]),
 'BayesianRidge':('Bayesian linear',[{}]),'ARD':('Bayesian linear',[{}]),
 'Huber':('Robust linear',[{'epsilon':1.35},{'epsilon':1.8}]),
 'PLS':('Latent linear',[{'n_components':3},{'n_components':6}]),
 'PCR':('Latent linear',[{'n_components':5},{'n_components':10}]),
 'SplineRidge':('Spline',[{'knots':3,'alpha':1},{'knots':5,'alpha':10}]),
 'KernelRidge':('Kernel',[{'alpha':.1,'gamma':.03},{'alpha':1,'gamma':.1}]),
 'SVR':('Kernel',[{'C':1,'gamma':'scale'},{'C':10,'gamma':'scale'}]),
 'KNN':('Neighbour',[{'n_neighbors':5},{'n_neighbors':15}]),
 'DecisionTree':('Tree',[{'max_depth':4},{'max_depth':8}]),
 'RandomForest':('Bagged trees',[{'max_depth':8},{'max_depth':None}]),
 'ExtraTrees':('Bagged trees',[{'max_depth':8},{'max_depth':None}]),
 'Bagging':('Bagged trees',[{'max_samples':.7},{'max_samples':1.}]),
 'GradientBoosting':('Boosted trees',[{'max_depth':2},{'max_depth':3}]),
 'HistGradientBoosting':('Boosted trees',[{'max_leaf_nodes':15},{'max_leaf_nodes':31}]),
 'AdaBoost':('Boosted trees',[{'learning_rate':.03},{'learning_rate':.1}]),
 'XGBoost':('Boosted trees',[{'max_depth':3},{'max_depth':5}]),
 'LightGBM':('Boosted trees',[{'num_leaves':15},{'num_leaves':31}]),
 'CatBoost':('Boosted trees',[{'depth':3},{'depth':5}]),
 'MLP':('Neural network',[{'hidden_layer_sizes':(16,)},{'hidden_layer_sizes':(32,)}])}
CONFIGS=[{'id':f'{method}_{i+1}','method':method,'family':family,'parameters':p}
         for method,(family,params) in METHODS.items() for i,p in enumerate(params)]
assert len(METHODS)==24

def estimator(c,seed):
    m=c['method'];p=c['parameters'].copy()
    if m=='OLS':return StableOLS()
    if m=='Ridge':return Ridge(**p)
    if m=='Lasso':return Lasso(max_iter=20000,tol=1e-5,**p)
    if m=='ElasticNet':return ElasticNet(max_iter=20000,tol=1e-5,**p)
    if m=='BayesianRidge':return BayesianRidge(max_iter=500)
    if m=='ARD':return ARDRegression(max_iter=500)
    if m=='Huber':return HuberRegressor(max_iter=2000,alpha=.001,**p)
    if m=='PLS':return PLSRegression(scale=False,max_iter=1000,**p)
    if m=='PCR':return make_pipeline(PCA(svd_solver='full',**p),StableOLS())
    if m=='SplineRidge':
        knots=p.pop('knots');return make_pipeline(SplineTransformer(n_knots=knots,degree=2,include_bias=False),StandardScaler(),Ridge(**p))
    if m=='KernelRidge':return KernelRidge(kernel='rbf',**p)
    if m=='SVR':return SVR(kernel='rbf',epsilon=.02,**p)
    if m=='KNN':return KNeighborsRegressor(weights='distance',**p)
    if m=='DecisionTree':return DecisionTreeRegressor(min_samples_leaf=3,random_state=seed,**p)
    if m=='RandomForest':return RandomForestRegressor(n_estimators=200,min_samples_leaf=3,n_jobs=1,random_state=seed,**p)
    if m=='ExtraTrees':return ExtraTreesRegressor(n_estimators=200,min_samples_leaf=3,n_jobs=1,random_state=seed,**p)
    if m=='Bagging':return BaggingRegressor(estimator=DecisionTreeRegressor(max_depth=8,min_samples_leaf=3),n_estimators=100,n_jobs=1,random_state=seed,**p)
    if m=='GradientBoosting':return GradientBoostingRegressor(n_estimators=200,learning_rate=.05,min_samples_leaf=3,random_state=seed,**p)
    if m=='HistGradientBoosting':return HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=seed,**p)
    if m=='AdaBoost':return AdaBoostRegressor(estimator=DecisionTreeRegressor(max_depth=3,min_samples_leaf=3),n_estimators=150,loss='linear',random_state=seed,**p)
    if m=='XGBoost':return XGBRegressor(n_estimators=250,learning_rate=.05,reg_lambda=1,n_jobs=1,random_state=seed,verbosity=0,**p)
    if m=='LightGBM':return LGBMRegressor(n_estimators=250,learning_rate=.05,reg_lambda=1,n_jobs=1,random_state=seed,verbosity=-1,**p)
    if m=='CatBoost':return CatBoostRegressor(iterations=300,learning_rate=.05,l2_leaf_reg=5,thread_count=1,random_seed=seed,verbose=False,allow_writing_files=False,**p)
    if m=='MLP':return MLPRegressor(alpha=.1,max_iter=1500,early_stopping=False,random_state=seed,**p)
    raise ValueError(m)

def topsis(matrix,weights=(1/3,1/3,1/3)):
    """Vector-normalized TOPSIS, all cost criteria. Degenerate columns carry no weight."""
    x=np.asarray(matrix,float);w=np.asarray(weights,float)
    if not np.isfinite(x).all():raise ValueError('Nonfinite TOPSIS criterion')
    active=(np.ptp(x,axis=0)>1e-12)&(w>0)
    if not active.any():return np.full(len(x),.5)
    x=x[:,active];w=w[active]/w[active].sum()
    v=x/np.linalg.norm(x,axis=0)*w
    dplus=np.linalg.norm(v-v.min(axis=0),axis=1)
    dminus=np.linalg.norm(v-v.max(axis=0),axis=1)
    return np.divide(dminus,dplus+dminus,out=np.full(len(x),.5),where=dplus+dminus>1e-15)
