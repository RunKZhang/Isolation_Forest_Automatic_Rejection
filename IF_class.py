import mne
import numpy as np
import mne_features
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

class if_reject:
    def __init__(self, epochs, selected_funcs):
        # variables used to initilization
        self.epochs = epochs
        self.selected_funcs = selected_funcs
        self.sfreq = epochs.info['sfreq']
        self.feature_transformed = None
        self.np_data = self.epochs.get_data()
        self.extract_feature()

        # variables used to control iteration
        self.drop_list = []
        self.keep_list = np.arange(0,self.np_data.shape[0],1).tolist()
        self.bet_class_dist = []
        self.iter_num = 0

        # variables used to calculate centroid
        self.feature_pca = None
        self.pca_reduction()
    
    def extract_feature(self):
        feature = mne_features.feature_extraction.extract_features(self.np_data, self.sfreq, self.selected_funcs)
        stand = StandardScaler()
        feature_transformed = stand.fit_transform(feature)
        self.feature_transformed = feature_transformed

    def pca_reduction(self):
        pca = PCA(n_components=1)
        self.feature_pca = pca.fit_transform(self.feature_transformed).squeeze()

    def iso_iteration(self):
        while(1):
    
            # Get retain_feature and build a mapping
            retain_feature = self.feature_transformed[self.keep_list]
            mapping = {}
            for i in range(0,len(self.keep_list)):
                mapping[i] = self.keep_list[i]
            
            feature_pca_keep = self.feature_pca[self.keep_list]
            feature_pca_drop = self.feature_pca[self.drop_list]
            
            if len(feature_pca_drop)==0:
                min_drop = 0
            else:
                min_drop = np.min(feature_pca_drop)
            max_keep = np.max(feature_pca_keep)
            
            self.bet_class_dist.append([self.iter_num, min_drop, max_keep, (min_drop-max_keep)**2])
    
            if self.iter_num>1 and self.bet_class_dist[self.iter_num][3] == self.bet_class_dist[self.iter_num-1][3]:
                 break
            else:
                # print(f'iteration num:{iter_num}, keep_list_len:{len(keep_list)}, drop_list_len:{len(drop_list)}')
                print(f'keep_list_len:{len(self.keep_list)}')
    
            # Use Isolation Forest from sklean 
            Iso = IsolationForest()
            pred = Iso.fit_predict(retain_feature)
    
            # Obtain indexes and features of goutliers and inliers
            index_neg1 = np.where(pred==-1)[0]
            index_pos1 = np.where(pred==1)[0]
            feature_neg1 = self.feature_pca[index_neg1]
            feature_pos1 = self.feature_pca[index_pos1]
            # print(f'index_neg1:{index_neg1}')

            # Calculate the mean of each outliers
            # mean_neg = np.mean(feature_neg1)
    
            # Calculate the mean of inliers, and it is used as mass of gravity to pull
            mean_center_pos = np.min(feature_pos1)
    
            # print(index_neg1)
            for j in range(0,len(index_neg1)):
                if feature_neg1[j]>mean_center_pos:
                # if mean_neg[j]>mean_center_pos:
                    idx = self.keep_list.index(mapping[index_neg1[j]])
                # print(f'index_neg1[j]:{index_neg1[j]}, index:{idx}')            
                    self.drop_list.append(self.keep_list[idx])
                    self.keep_list.pop(idx)
            
            self.iter_num+=1

    def run(self):
        self.iso_iteration()
        epoch_copy = self.epochs.copy().drop(self.drop_list)
        return epoch_copy
    
    def get_keep_list(self):
        return self.keep_list
    
    def get_drop_list(self):
        return self.drop_list
    
    def get_iter_num(self):
        return self.iter_num
    
    def get_bet_class_list(self):
        return self.bet_class_dist