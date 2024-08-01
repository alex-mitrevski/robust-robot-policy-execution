import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ipdb
import numpy as np
from sklearn.cluster import KMeans
from scipy.spatial import distance
# Load the features
from sklearn.neighbors import NearestNeighbors
import os
import json
import argparse
def get_train_mean_std(nominal_features):
    nominal_mean = torch.mean(nominal_features,dim=0)
    nominal_std = torch.std(nominal_features,dim=0)
    return nominal_mean, nominal_std

def normalize_features(features,mean,std):
    normalized_features=(features - mean) / (std + 1e-8) 
    normalized_features_np = normalized_features.numpy()
    return normalized_features

def extract_features(features_path):
    features = torch.load(features_path) 
    features_tensor = torch.stack(features).squeeze().cpu()
    return features_tensor

def visualize_features (nominal_features_path, anomalous_features_path,test_frames_json,save_path):
    with open(test_frames_json) as f:
        test_frames = json.load(f)
    title= anomalous_features_path.split('/')[-1].split('.')[0]
    anom_frames= test_frames[title]
    if len(anom_frames)!= 0:
        anom1_start,anom1_end = anom_frames[0],anom_frames[1]
        if len(anom_frames)>2:
            anom2_start,anom2_end = anom_frames[2],anom_frames[3]
        else:
            anom2_start,anom2_end = None,None
    else:
        anom1_start,anom1_end = None,None
        anom2_start,anom2_end = None,None
    
    threshold = test_frames['threshold']
    
    nom_features_tensor = extract_features(nominal_features_path)
    nominal_mean,nominal_std = get_train_mean_std(nom_features_tensor)
    nom_features_np= normalize_features(nom_features_tensor, nominal_mean,nominal_std)

    #load anomalous features
    anomalous_features_tensor= extract_features(anomalous_features_path)
    anomalous_features_np= normalize_features(anomalous_features_tensor,nominal_mean,nominal_std)

    #compute the nearest neighbour
    # Initialize the NearestNeighbors model
    k = 5  # Number of neighbors to use for KNN
    nn_model = NearestNeighbors(n_neighbors=k)


    # Fit the model on the nominal features
    nn_model.fit(nom_features_np) #nom_features or normalized_nominal_features

    # Find the distance to and index of the k-th nearest neighbors in the nominal data for each test feature
    distances, indices = nn_model.kneighbors(anomalous_features_np) #anomalous_features  or normalized_anomalous_features

    # Use the distance to the k-th nearest neighbor as the anomaly score
    anomaly_scores = distances[:, 0]
    # ipdb.set_trace()
    #plot anomaly scores
    title = anomalous_features_path.split('/')[-1].split('.')[0]
    plot_scores(anomaly_scores,title, anom1_start,anom1_end,anom2_start,anom2_end,threshold,save_path)


def plot_scores(anomaly_scores,title,anom1_start,anom1_end, anom2_start, anom2_end,threshold,save_path):
    #plot anomalous scores in the end
    # Plotting the anomaly scores
    indices = range(len(anomaly_scores))
    plt.figure(figsize=(10, 6))
    plt.scatter(indices, anomaly_scores, color='red', alpha=0.7, marker='o', label='Anomaly Score')

    plt.title('anomaly scores for test data: '+title)
    plt.xlabel('Test image frames')
    plt.ylabel('Anomaly Score (Distance to the nearest neighbour)')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.ylim(5, 55)
    # Optionally, you can also draw a horizontal line representing the threshold to visually identify outliers
    threshold = np.percentile(anomaly_scores, 90)  # Example: setting threshold at the 95th percentile
    plt.axhline(y=25, color='blue', linestyle='-', label='Threshold')
    plt.legend()

    # Adding dotted vertical lines at indices 79 and 115
    if anom1_start is not None and anom1_end is not None:
        plt.axvline(x=anom1_start, color='red', linestyle=':', linewidth=2, label='Index 79') #36,60 for 2nd dataset , 79 for 1st
        plt.axvline(x=anom1_end, color='red', linestyle=':', linewidth=2, label='Index 115') #43,107 for 2nd dataset, 115 for 1st
    
    if anom2_start is not None and anom2_end is not None:
        plt.axvline(x=anom2_start, color='red', linestyle=':', linewidth=2, label='Index 79') #36,60 for 2nd dataset , 79 for 1st
        plt.axvline(x=anom2_end, color='red', linestyle=':', linewidth=2, label='Index 115') #43,107 for 2nd dataset, 115 for 1st
    # plt.show()
    plt.savefig(os.path.join(save_path,title+'.png'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--nominal_features_path', type=str, required=True)
    parser.add_argument('--anomalous_features_path', type=str, required=True)
    parser.add_argument('--test_frames_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()
    for anomaly in os.listdir(args.anomalous_features_path):
        anomaly_path = os.path.join(args.anomalous_features_path,anomaly)
        visualize_features(args.nominal_features_path, anomaly_path,args.test_frames_path,args.save_path)


if __name__ == '__main__':
    main()



