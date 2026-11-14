"""SVM model selection and repeated k-means experiments for HW7."""
import numpy as np
import pandas as pd
from sklearn.svm import SVC, LinearSVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

SEED=552
LABELS=["Family","Genus","Species"]

def scores(y,p):
    match=np.asarray(y)==np.asarray(p)
    count=match.sum(axis=1)
    return dict(exact_match=np.all(match,axis=1).mean(),
        mean_hamming_distance=(3-count).mean(),hamming_loss=1-match.mean(),
        hamming_score=match.mean(),jaccard_score=np.mean(count/(6-count)))

def tune_models(X,y,kind,output):
    models=[]; records=[]
    for j,name in enumerate(LABELS):
        if kind=="RBF":
            model=OneVsRestClassifier(SVC(kernel="rbf",cache_size=160),n_jobs=1)
            grid={"estimator__C":[1,10,100],"estimator__gamma":[.1,1,10]}
        else:
            model=Pipeline([("scale",StandardScaler()),("svm",LinearSVC(
                penalty="l1",loss="squared_hinge",dual=False,max_iter=30000,tol=1e-4,
                class_weight="balanced" if kind=="L1 balanced" else None,
                random_state=SEED))])
            grid={"svm__C":[.01,.1,1,10,100]}
        cv=StratifiedKFold(10,shuffle=True,random_state=SEED)
        gs=GridSearchCV(model,grid,cv=cv,scoring="accuracy",n_jobs=3,
                        return_train_score=False,error_score="raise")
        with parallel_config(backend="loky",inner_max_num_threads=1):
            gs.fit(X,y[:,j])
        pd.DataFrame(gs.cv_results_).to_csv(output/(kind.replace(" ","_")+"_"+name+"_cv.csv"),index=False)
        best=gs.best_params_
        C=best.get("estimator__C",best.get("svm__C"))
        gamma=best.get("estimator__gamma",np.nan)
        records.append(dict(model=kind,label=name,C=C,gamma=gamma,
            sigma=np.sqrt(1/(2*gamma)),cv_accuracy=gs.best_score_))
        models.append(gs.best_estimator_)
    return models,records

def kmeans_trial(X,y,run):
    with threadpool_limits(limits=1):
        best=None; scans=[]
        total=((X-X.mean(axis=0))**2).sum()
        for k in range(1,51):
            km=KMeans(n_clusters=k,n_init=3,random_state=SEED+run,
                      algorithm="lloyd",max_iter=300).fit(X)
            ch=np.nan if k==1 else calinski_harabasz_score(X,km.labels_)
            scans.append(dict(run=run+1,k=k,CH=ch,inertia=km.inertia_,iterations=km.n_iter_))
            if k>1 and (best is None or ch>best[0]):
                best=(ch,k,km.labels_.copy())
        ch,k,cluster=best
        pred=np.empty_like(y); rows=[]
        for group in range(k):
            mask=cluster==group; triple=[]
            for j in range(3):
                values,counts=np.unique(y[mask,j],return_counts=True)
                triple.append(values[counts.argmax()])
            pred[mask]=triple
            rows.append(dict(run=run+1,cluster=group,size=int(mask.sum()),
                             **dict(zip(LABELS,triple))))
        return dict(run=run+1,k=k,CH=ch,**scores(y,pred)),scans,rows

def monte_carlo(X,y,output):
    with parallel_config(backend="loky",inner_max_num_threads=1):
        runs=Parallel(n_jobs=3)(delayed(kmeans_trial)(X,y,r) for r in range(50))
    result=pd.DataFrame([r[0] for r in runs])
    scans=pd.DataFrame([v for r in runs for v in r[1]])
    majorities=pd.DataFrame([v for r in runs for v in r[2]])
    result.to_csv(output/"kmeans_50_runs.csv",index=False)
    scans.to_csv(output/"kmeans_all_candidates.csv",index=False)
    majorities.to_csv(output/"kmeans_cluster_majorities.csv",index=False)
    return result,scans,majorities

class MulticlassChain:
    """Family -> Genus -> Species with one-hot preceding labels."""
    def __init__(self,parameters):self.parameters=parameters
    def fit(self,X,y):
        self.models=[];self.encoders=[];A=X.copy()
        for j,row in enumerate(self.parameters):
            m=OneVsRestClassifier(SVC(C=row["C"],gamma=row["gamma"],kernel="rbf"),n_jobs=1).fit(A,y[:,j])
            self.models.append(m)
            enc=OneHotEncoder(handle_unknown="ignore",sparse_output=False).fit(y[:,j:j+1])
            self.encoders.append(enc);A=np.column_stack([A,enc.transform(y[:,j:j+1])])
        return self
    def predict(self,X):
        A=X.copy();p=[]
        for m,enc in zip(self.models,self.encoders):
            label=m.predict(A);p.append(label)
            A=np.column_stack([A,enc.transform(label[:,None])])
        return np.column_stack(p)
