from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.covariance import EllipticEnvelope
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def z_score_outliers(data, pil_image, bboxes, threshold=3, output_dir='z_score.png'):
    z_scores = stats.zscore(data)
    outliers = np.where(np.abs(z_scores) > threshold)
    ypred = np.ones(data.shape)
    ypred[outliers] = -1

    plot_results(data, outliers[0], 'Z-Score', output_dir)
    boxes = plot_bbox(ypred, bboxes, pil_image, output_dir)
    return boxes

def iqr_outliers(data, pil_image, bboxes, output_dir='iqr.png'):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = np.where((data < lower_bound) | (data > upper_bound))

    ypred = np.ones(data.shape)
    ypred[outliers] = -1

    plot_results(data, outliers[0], 'IQR', output_dir)
    boxes = plot_bbox(ypred, bboxes, pil_image, output_dir)
    return boxes

def rob_cov(data, pil_image, bboxes, threshold=0.25, output_dir='robust_cov.png'):
    robust_cov = EllipticEnvelope(contamination=threshold)
    y_pred_cov = robust_cov.fit_predict(data)
    visualize_results_with_legend(data, y_pred_cov, 'Robust Covariance', output_dir)
    boxes = plot_bbox(y_pred_cov, bboxes, pil_image, output_dir)
    return boxes

def svm_outliers(data, pil_image, bboxes, threshold=0.1, output_dir='svm.png'):
    one_class_svm = OneClassSVM(nu=threshold, kernel="rbf", gamma=0.1)
    y_pred_svm = one_class_svm.fit_predict(data)
    visualize_results_with_legend(data, y_pred_svm, 'One Class SVM', output_dir)
    boxes = plot_bbox(y_pred_svm, bboxes, pil_image, output_dir)
    return boxes

def svm_sgd_outliers(data, pil_image, bboxes, threshold=0.1, output_dir='svm_sgd.png'):
    one_class_svm = OneClassSVM(nu=threshold, kernel="linear")
    y_pred_svm = one_class_svm.fit_predict(data)
    visualize_results_with_legend(data, y_pred_svm, 'One Class SVM', output_dir)
    boxes = plot_bbox(y_pred_svm, bboxes, pil_image, output_dir)
    return boxes

def isolation_forest_outliers(data, pil_image, bboxes, threshold=0.25, output_dir='isolation_forest.png'):
    isolation_forest = IsolationForest(contamination=threshold,random_state=42)
    y_pred_iso = isolation_forest.fit_predict(data)
    visualize_results_with_legend(data, y_pred_iso, 'Isolation Forest', output_dir)
    boxes = plot_bbox(y_pred_iso, bboxes, pil_image, output_dir)
    return boxes

def lof_outliers(data, pil_image, bboxes, threshold=0.25, output_dir='lof.png'):
    lof = LocalOutlierFactor(n_neighbors=20, contamination=threshold)
    y_pred_lof = lof.fit_predict(data)
    visualize_results_with_legend(data, y_pred_lof, 'Local Outlier Factor', output_dir)
    boxes = plot_bbox(y_pred_lof, bboxes, pil_image, output_dir)
    return boxes

# Function to visualize the results with a legend
def visualize_results_with_legend(data, y_pred, title, output_dir):
    plt.figure(figsize=(10, 6))
    outliers = y_pred == -1
    inliers = y_pred == 1
    plt.scatter(np.where(inliers)[0], data[inliers], c='blue', label='Inliers')
    plt.scatter(np.where(outliers)[0], data[outliers], c='red', label='Outliers')
    plt.title(title)
    plt.grid(True)
    plt.xlabel('Index')
    plt.ylabel('Area (px sq.)')
    plt.legend()
    plt.savefig(output_dir)
    plt.close()

# Function to plot results
def plot_results(data, outliers_idx, title, output_dir):
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(data)), data, c='blue', label='inliers')
    plt.scatter(outliers_idx, data[outliers_idx], c='red', label='Outliers')
    plt.grid(True)
    plt.title(title)
    plt.xlabel('Index')
    plt.ylabel('Area (px sq.)')
    plt.legend()
    plt.savefig(output_dir)
    plt.close()

def plot_bbox(ypred, bboxes, pil_image, output_dir):
    # define new output_dir: output_dir= '.../example.png' -> output_dir = '.../example_filtered.png'
    output_dir = output_dir.split('.')[0] + '_filtered.jpg'
    # Filter out outliers from the original bboxes list based on the calculated threshold
    filtered_bboxes = [bbox for bbox, label in zip(bboxes, ypred) if (label == 1)]
    # Plotting the bounding boxes directly on the array converted to an image, without saving and re-reading it
    fig, ax = plt.subplots(figsize=(10, 10))
    # Directly display the array as an image
    ax.imshow(pil_image, cmap='gray')

    # Plotting the bounding boxes on the directly converted image
    for bbox in filtered_bboxes:
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.savefig(output_dir)

    return filtered_bboxes