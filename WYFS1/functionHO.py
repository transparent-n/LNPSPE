import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
# error rate
def error_rate(feature, label, x, opts):
    # parameters
    k     = opts['k']
    coefficients=opts['coefficients']
    feature_cols=opts['feature_cols']
    # # fold  = opts['fold']
    # xt    = fold['xt']
    # yt    = fold['yt']
    # xv    = fold['xv']
    # yv    = fold['yv']
    

    # Define selected features
    selected_features = feature[:, x == 1]
    selected_feature_names = np.array(feature_cols)[np.where(x == 1)]
    
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from imbens.ensemble import SelfPacedEnsembleClassifier
    clf= LGBMClassifier(random_state=100 )
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
        matthews_corrcoef
    X_train, X_test, Y_train, Y_test =train_test_split(selected_features, label, test_size=0.2,random_state=100,stratify=label)
    clf.fit(X_train, Y_train, )  # sample_weight=weights
    Y_pred = clf.predict(X_test)
    acc = accuracy_score(Y_test, Y_pred)
    error = acc
    print(selected_feature_names)
    n=len(selected_feature_names)
    weighted_error = (np.sum(np.sqrt([coefficients.get(col, 0) * 0.01 for col in selected_feature_names])))
    return error, weighted_error


# Error rate & Feature size
def Fun(feature, label, x, opts):
    # Parameters
    lambda_value=opts['lambda_value']
    alpha    = 0.99
    beta     = 1 - alpha
    # Original feature size
    max_feat = len(x)
    # Number of selected features
    num_feat = np.sum(x == 1)
    # Solve if no feature selected
    if num_feat == 0:
        cost  = 1
    else:
        # Get error rate
        error,weighted_error = error_rate(feature, label, x, opts)
        # Objective function
        penalty = lambda_value * weighted_error
        cost =alpha *( np.exp(-(error + penalty)))+ beta * (np.exp(-(1/num_feat)))
        # cost  = alpha * error + beta * (num_feat / max_feat)
        # print('num_feat',num_feat)
        print('error',error,'weighted_error',weighted_error,num_feat)
        print('cost', cost)
        # print('cost',cost)
        
    return cost#error



