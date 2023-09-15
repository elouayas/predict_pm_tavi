import pandas as pd
pd.set_option('display.max_columns', 500)
from sklearn.preprocessing import StandardScaler,MinMaxScaler
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold,train_test_split,PredefinedSplit,GridSearchCV,KFold
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import RandomOverSampler,SMOTE 
from sklearn.metrics import roc_auc_score,balanced_accuracy_score,accuracy_score,f1_score,confusion_matrix,precision_score,recall_score,average_precision_score,roc_curve,precision_recall_curve
import scipy.stats as st
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import shap
from sklearn.linear_model import Lasso
from sklearn.impute import KNNImputer

per_cols = ['localisation_Bloc opératoire','localisation_Cardio interventionnelle','localisation_Salle hybride','localisation_nan','abord_Carotidien','abord_Ilio fémoral','abord_Trans aortique','abord_Transapical','marque_ACURATE','marque_COREVALVE','marque_EDWARDS','modele_PORTICO','taille_valve_23','taille_valve_25','taille_valve_26','taille_valve_27','taille_valve_29','taille_valve_34','predilatation_0','predilatation_1','postdilatation_0','postdilatation_1','nombre_cardiologues_interventionnels','nombre_chirurgiens','nb_valve_implantée','prof_impl_valv','delta_msid','delta_msid_d','nominal','oversizing','oversizing_2','oversizing_3','oversizing_4','AVA_PNA_pre_ratio','modele_EVOLUT_PRO','modele_EVOLUT_R','modele_SAPIEN_III']

post_cols = ['tsv_post_Non','tsv_post_Oui','fuite_aortique_Aucune','fuite_aortique_Grade 1','fuite_aortique_Grade 2','fuite_aortique_Grade 3','fuite_aortique_Grade 4','fuite_aortique_nan','localisation_fuite_Centrale','localisation_fuite_Centrale et péri-prothétique','localisation_fuite_Péri-prothétique','localisation_fuite_nan','fuite_mitrale_Aucune','fuite_mitrale_Grade 1','fuite_mitrale_Grade 2','fuite_mitrale_Grade 3','fuite_mitrale_Grade 4','fuite_mitrale_nan','bloc_branche_post_bbd','bloc_branche_post_bbg','bloc_branche_post_non','pr_j0','qrs_j0','gradient_moyen_post_tavi','surface_valvulaire_post_tavi','fraction_ejection_post_tavi','valeur_paps_post','delta_pr','delta_qrs','AVA_ratio','AVA_PNA_post_ratio']
    
    
def grid_model(model_name):
    if model_name=='lgb':
        clf = LGBMClassifier(n_jobs=1,verbose=-1)
        param_grid = {'max_depth': [-1],
                     'boosting_type':['dart'],
                        'num_leaves':[40]}
    if model_name=='svc':
        clf = SVC(probability=True)
        param_grid = {'kernel': ['linear']}
    if model_name =='rfo':
        clf = RandomForestClassifier(n_jobs=1)
        param_grid = {'max_depth': [7,10]}        
    if model_name =='xgb':
        clf = XGBClassifier(n_jobs=1)
        param_grid = {'max_depth': [5,7,10]}
    if model_name =='his':
        clf = HistGradientBoostingClassifier()
        param_grid = {'max_depth': [7,10]}

    if model_name =='lre':
        clf = LogisticRegression(max_iter=500)
        param_grid = {'C':[100,1.0, 0.1]}
    if model_name =='mlp':
        clf = MLPClassifier(early_stopping=True,max_iter=300)
        param_grid = {'hidden_layer_sizes' :[(16,),(32,),(16,16),(32,32)]}
    
    if model_name=='dtr':
        clf = DecisionTreeClassifier()
        param_grid = {'max_depth': [5,7],'min_samples_leaf':[30]}
    
    if model_name =='gnb':
        clf =GaussianNB()
        param_grid = {}
    
    return clf, param_grid

def compute_pds(X_,y_):
    X_train, X_val, y_train, y_val = train_test_split(X_, y_, train_size = 0.7, stratify = y_,random_state=42)
    split_index = [-1 if x in X_train.index else 0 for x in X_.index]
    pds = PredefinedSplit(test_fold = split_index)
    return pds


def normalize(data,type_stand = 'standard'):
    df_ = data.copy()
    if type_stand =='standard':
        scaler=StandardScaler()
    if type_stand=='minmax':
        scaler = MinMaxScaler()
    scaled_data =  scaler.fit_transform(df_)
    df_scaled = pd.DataFrame(columns=df_.columns,data=scaled_data)
    return scaled_data,df_scaled,scaler

def cut_outliers(data,test_min,test_max,test = False):
    df_ = data.copy()
    if not test :
        scaled_data,df_scaled,fitted_scaler = normalize(df_,type_stand = 'standard')
        min_ = fitted_scaler.inverse_transform(np.expand_dims(np.array([-3]*df_scaled.shape[1]),0))[0]
        max_ = fitted_scaler.inverse_transform(np.expand_dims(np.array([3]*df_scaled.shape[1]),0))[0]
    else :
        min_ = test_min
        max_ = test_max
    cols = df_.columns
    for i in range(len(cols)):
        df_[cols[i]] = df_[cols[i]].clip(min_[i],max_[i])
    return df_,min_,max_

def value_to_impute(data,type_impute='median'):
    df_ = data.copy()
    if type_impute =='mode':
        dict_impute = df_.mode(axis=0, numeric_only=False).mean().to_dict()
    if type_impute =='mean': 
        dict_impute = df_.mean().to_dict()
    if type_impute =='median': 
        dict_impute = df_.median().to_dict()
    if 'qrs_j0' in df_.columns:
        dict_impute['qrs_j0'] = df_['qrs_j0'].mode().mean()
    if 'delta_qrs' in df_.columns:
        dict_impute['delta_qrs'] = df_['delta_qrs'].mode().mean()
    if 'pr_j0' in df_.columns:
        dict_impute['pr_j0'] = df_['pr_j0'].mode().mean()
    return dict_impute

def impute_missing_values(df_,dict_impute):
    return df_.fillna(dict_impute)





def run_stats(scores, name=" "):
    if len(scores) == 1:
        low, up = 0., 1.
    elif len(scores) < 30:
        low, up = st.t.interval(0.95, df = len(scores) - 1, loc = np.mean(scores), scale = st.sem(scores))
    else:
        low, up = st.norm.interval(0.95, loc = np.mean(scores), scale = st.sem(scores))
    mean_ = np.mean(scores)
    std_ = np.std(scores)
    min_ = np.min(scores)
    max_ = np.max(scores)
    #print("{:s} {:.2f} (± {:.2f}) (conf: [{:.2f}, {:.2f}]) (worst: {:.2f}, best: {:.2f})".format(name, 100 * mean_ , 100 * std_, 100 * low, 100 * up, 100 * min_, 100 * max_))
    return mean_,std_,low,up,min_,max_

def resample(X__,y__,type_resampling_):
    X_ = X__.copy()
    y_ = y__.copy()
    if type_resampling_=='ros':
        ros = RandomOverSampler(random_state=42)
        X_, y_ = ros.fit_resample(X_, y_)
    if type_resampling_=='smote':
        smote = SMOTE(random_state=42)
        X_, y_ = smote.fit_resample(X_, y_)
    return X_,y_

def drop_pre_per_post(X__,pre_per_post_):
    X_ = X__.copy()


    if pre_per_post_=='pre':
        cols = X_.columns 
        remove_cols = [col for col in cols if col in per_cols+post_cols]
        
        X_ = X_.drop(columns = remove_cols)
    if pre_per_post_=='per':
        cols = X_.columns 
        remove_cols = [col for col in cols if col in post_cols]
        X_ = X_.drop(columns = remove_cols)
    return X_


def correct_columns(rescale_cols,encode_cols,drop_type,pre_per_post_):
    res_cols = rescale_cols
    enc_cols = encode_cols
    
    if pre_per_post_== 'pre':
        res_cols_ = [c for c in res_cols if c not in per_cols+post_cols]
        enc_cols_ = [c for c in enc_cols if c not in per_cols+post_cols]
    elif pre_per_post_== 'per':
        res_cols_ = [c for c in res_cols if c not in post_cols]
        enc_cols_ = [c for c in enc_cols if c not in post_cols]
    else :
        res_cols_ = res_cols
        enc_cols_ = enc_cols
    return res_cols_,enc_cols_

def takeClosest(num,collection):
    return min(collection,key=lambda x:abs(x-num))

dict_rename={'AVA_PNA_post_ratio':'TTE post-TAVI AVA-PNA ratio',
'AVA_ratio':'TTE AVA ratio',
 'MS_d':'dMS',
 'MS_s':'sMS',
 'abord_Trans aortique':'Trans-aortic access',
 'age':'Age',
 'ait_avc_Non':'No previous stroke',
 'ait_avc_Oui':'Previous stroke',
 'aomi_Non':'No prior peripheral arterial disease',
 'aomi_Oui':'Prior peripheral arterial disease',
 'bioprothese_aortique_Oui':'Prior surgical aortic bioprosthesis',
 'bloc_branche_post_bbd':'Post-TAVI RBBB',
 'bloc_branche_post_non':'No post-TAVI BBB',
 'bloc_branche_pre_bbd':'Pre-TAVI RBBB',
 'bloc_branche_pre_bbg':'Pre-TAVI LBBB',
 'calc_risque_n':'Risk zone calcification',
 'calc_septum_n':'Calcium in basal septum',
 'calcif_tiers_sup_aorte_0':'No aortic arch calcification',
 'calcif_tiers_sup_aorte_1':'Aortic arch calcification',
 'clairance':'Creatinine clearance',
 'creatinine_lors_inclusion':'Creatinine', 
 'delta_msid':'ΔsMSID',
 'delta_msid_d':'ΔdMSID',
 'delta_qrs':'ΔQRS',
 'diff_MS':'ΔdsMS',
 'euroscore_logistique':'Logistic euroSCORE',
 'fraction_ejection_post_tavi':'Post-TAVI LVEF',
 'fuite_aortique_Aucune':'No post-TAVI AR',
 'fuite_aortique_nan':'Post TAVI AR NaN',
 'fuite_aortique_pretavi_Grade 1':'Pre-tavi AR I',
 'fuite_mitrale_Aucune':'No post-TAVI MR',
 'fuite_mitrale_nan':'Post-TAVI MR NaN',
 'fuite_mitrale_pretavi_Aucune':'No pre-TAVI MR',
 'fuite_mitrale_pretavi_nan':'Pre-TAVI MR NaN',
 'imc':'BMI',
 'insuffisance_renale_chronique_3':'Chronic kidney disease 3',
 'insuffisance_renale_chronique_4':'Chronic kidney disease 4',
 'insuffisance_renale_chronique_Non':'No chronic kidney disease',
'lcc_calc_1.0':'Mild LCC calcification',
 'lcc_calc_2.0':'Moderate LCC calcification',
 'lcc_calc_n':'LCC calcification',
 'localisation_fuite_Centrale et péri-prothétique':'Peri and central-prosthetic AR',
 'localisation_fuite_Péri-prothétique':'AR localisation',
 'localisation_fuite_nan':'AR localisation NaN',
 'lvot_calc_0.0':'No LVOT calcification',
 'lvot_calc_1.0':'LVOT calcification',
 'marque_ACURATE':'ACURATE',
 'marque_COREVALVE':'COREVALVE',
 'modele_NEO_M':'ACURATE Neo M',
 'modele_NEO_S':'ACURATE Neo S',
 'ncc_calcif_1.0':'Mild NCC calcification',
 'ncc_calcif_2.0':'Moderate NCC calcification',
 'ncc_calcif_n':'NCC calcification',
 'nombre_cardiologues_interventionnels':'Number of interventional cardiologists',
 'nominal':'Prosthesis nominal area',
 'nyha34_0':'No NYHA 3&4',
 'nyha34_1':'NYHA 3&4',
 'petit_diametre_anneau':'Minimal CT aortic annulus diameter',
 'poids':'Weight',
 'pr_j0':'Post-TAVI PR',
 'pr_pre_tavi':'Pre-TAVI PR',
 'qrs_j0':'Post-TAVI QRS',
 'qrs_pre_tavi':'Pre-TAVI QRS',
 'score_calcique':'Aortic calcium score',
 'sexe_Femme':'Women',
 'sexe_Homme':'Men',
 'si_oui_insuline_Non':'Diabetes with insuline use',
 'si_oui_nyha_II':'NYHA 2',
 'si_oui_nyha_III':'NYHA 3',
 'surface_systole':'Systolic CT area',
 'surface_valvulaire_post_tavi':'Post-TAVI TTE valve area',
 'surface_valvulaire_pre_tavi':'Pre-TAVI TTE valve area',
 'syncope_Oui':'Syncope',
 'taille':'Height',
 'valeur_paps_post':'Post-TAVI PASP',
 'marque_EDWARDS':'EDWARDS',
 'oversizing_4':'Oversizing',
 'taille_valve_29':'Valve size 29',
 'taille_valve_26':'Valve size 26',
 'modele_EVOLUT_R':'EVOLUT R'            }



def compute_metrics(true,pred,scores_positives):
    balanced_accuracy = balanced_accuracy_score(true, pred)
    accuracy = accuracy_score(true, pred)
    f1 = f1_score(true, pred)
    auc = roc_auc_score(true,scores_positives)
    precision = precision_score(true,pred)
    recall = recall_score(true,pred)
    matrix_conf = confusion_matrix(true,pred)
    return balanced_accuracy,accuracy,f1,auc,precision,recall,matrix_conf



def train_model(df__,to_rescale,to_encode,target_name,normalize_type,cut_outliers_bool,
                impute_missing,model_name_,per_post_pre,morpho,type_resampling,
                n_runs=10,n_folds=10,n_folds_cv=10,run_shap=False,curves=False):
    
    X = df__[to_encode+to_rescale].copy()
    y = df__[target_name].copy()
    to_rescale_,to_encode_ = correct_columns(to_rescale,to_encode,morpho,per_post_pre) # enlever ici si useless

    return_list = []
    ##METRICS##
    balanced_accuracy_list,accuracy_list,f1_list,auc_list,matrix_conf_list,precision_list,recall_list  = [],[],[],[],[],[],[]
    list_df_shap = []
    mean_tprs_list = []
    base_fpr = np.linspace(0, 1, 101)
    y_test_list = []
    y_pred_list = []
    y_score_list =  []
    prec_list =  []
    rec_list =  []
    ##METRICS##
    
    for j in np.random.randint(0, high=10000, size=n_runs): #10
        ##METRICS##
        tprs = []
        y_test_list_ =  []
        y_score_list_ =  []
        ##METRICS##
        skf = StratifiedKFold(n_splits=n_folds,shuffle=True, random_state=j) #10
        
        for train_index, test_index in skf.split(X, y):
            X_train, X_test, y_train, y_test = X.loc[train_index].reset_index(drop=True), X.loc[test_index].reset_index(drop=True), y.loc[train_index].reset_index(drop=True),y.loc[test_index].reset_index(drop=True) 
            if impute_missing:
                if impute_missing=='iterative' or impute_missing=='knn':
                    if impute_missing=='iterative': 
                        imp = IterativeImputer(max_iter=10, random_state=0)
                    else : 
                        imp = KNNImputer(n_neighbors=5)
                    X_train = imp.fit_transform(X_train)
                    X_test = imp.transform(X_test)
                    X_train =pd.DataFrame(columns = X.columns,data=X_train)
                    X_test =pd.DataFrame(columns = X.columns,data=X_test)
                else : 
                    dict_impute = value_to_impute(X_train,impute_missing)
                    X_train = impute_missing_values(X_train,dict_impute)
                    X_test = impute_missing_values(X_test,dict_impute)     
            if cut_outliers_bool:
                X_train_cut,min_,max_ = cut_outliers(X_train[to_rescale_],None,None,False)
                X_test_cut,min_,max_ = cut_outliers(X_test[to_rescale_],min_,max_,True)
                X_train = pd.concat([X_train_cut,X_train[to_encode_]],axis=1)
                X_test = pd.concat([X_test_cut,X_test[to_encode_]],axis=1)
            if type_resampling:
                X_train,y_train = resample(X_train,y_train,type_resampling)
            if normalize_type : 
                _,X_train_scaled,fitted_scaler = normalize(X_train[to_rescale_],type_stand=normalize_type)
                X_test_ = fitted_scaler.transform(X_test[to_rescale_])
                X_test_scaled = pd.DataFrame(columns=to_rescale_,data=X_test_)
                X_train = pd.concat([X_train_scaled,X_train[to_encode_]],axis=1)
                X_test = pd.concat([X_test_scaled,X_test[to_encode_]],axis=1)
    
            clf, param_grid = grid_model(model_name_)
            g_model = GridSearchCV(estimator = clf,cv=StratifiedKFold(n_splits=n_folds_cv,shuffle=True, random_state=10), #10
                                   param_grid=param_grid,n_jobs=-1,scoring='roc_auc')
            g_model.fit(X_train, y_train)
            model =  g_model.best_estimator_
            model.fit(X_train, y_train)
            
            ###METRICS###
            balanced_accuracy,accuracy,f1,auc,precision,recall,matrix_conf= compute_metrics(y_test,model.predict(X_test),model.predict_proba(X_test)[:,1])    
            balanced_accuracy_list.append(balanced_accuracy)
            accuracy_list.append(accuracy)
            f1_list.append(f1)
            auc_list.append(auc)
            precision_list.append(precision)
            recall_list.append(recall)
            matrix_conf_list.append(matrix_conf)
            
            if run_shap:
                #shap#
                explainer = shap.Explainer(model.predict, X_test)
                shap_test = explainer(X_test)
                df_shap = pd.DataFrame(shap_test.values, columns = X_test.columns)
                list_df_shap.append(df_shap)
            #curvres
            if curves:
                y_score = model.predict_proba(X_test)
                fpr, tpr, _ = roc_curve(y_test, y_score[:, 1])
                y_test_list.append(y_test)
                y_score_list.append(y_score)
                y_test_list_.append(y_test)
                y_score_list_.append(y_score)
                
                tpr_intercep = np.interp(base_fpr, fpr, tpr)
                tpr_intercep[0] = 0.0
                tprs.append(tpr_intercep)
        
        ###METRICS###
        if curves : 
            prec, rec, _ = precision_recall_curve(np.concatenate(y_test_list_), np.concatenate(y_score_list_)[:, 1])
            prec_list.append(prec)
            rec_list.append(rec)
            tprs_ = np.array(tprs)
            mean_tprs = tprs_.mean(axis=0)
            mean_tprs_list.append(mean_tprs)
    if curves: 
        mean_tprs_ = np.array(mean_tprs_list)
        mean_tprs_array = mean_tprs_.mean(axis=0)
        std = mean_tprs_.std(axis=0)
        mean_prec, mean_rec, _ = precision_recall_curve(np.concatenate(y_test_list), np.concatenate(y_score_list)[:, 1])
        tprs_upper = np.minimum(mean_tprs_array + std, 1)
        tprs_lower = mean_tprs_array - std        
        AP = average_precision_score(np.concatenate(y_test_list), np.concatenate(y_score_list)[:, 1])
        interp_prec = []
        for r,p in zip(rec_list,prec_list):
            interp_prec.append([p[np.where(r==takeClosest(x,r))[0][0]] for x in mean_rec])
        std_prec = np.array(interp_prec).std(axis=0)
        prec_upper = np.minimum(mean_prec + std_prec, 1)
        prec_lower = mean_prec - std_prec 
    #print('auc, acc, f1 : ',round(np.mean(auc_list),2),round(np.mean(accuracy_list),2),round(np.mean(f1_list),2))
    
    for item in [balanced_accuracy_list,accuracy_list,f1_list,auc_list,precision_list,recall_list,matrix_conf_list]:
        return_list.append(item)
    
    if curves:
        for item in [ mean_rec,mean_prec,rec_list,prec_list,AP,std_prec,prec_upper,prec_lower,
                      mean_tprs_array,mean_tprs_list,base_fpr,tprs_lower,tprs_upper] :
            return_list.append(item)
    if run_shap :
        return_list.append(list_df_shap)
    
    
    return tuple(return_list)

#balanced_accuracy_list,accuracy_list,f1_list,auc_list,precision_list,recall_list,matrix_conf_list
#mean_rec,mean_prec,rec_list,prec_list,AP,std_prec,prec_upper,prec_lower,mean_tprs_array,mean_tprs_list,base_fpr,tprs_lower,tprs_upper
#list_df_shap