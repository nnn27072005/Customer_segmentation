"""
Customer Segmentation Clustering Library
"""
import datetime as dt
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from scipy import stats
from scipy.stats import boxcox
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

class DataCleaner:
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.df_uk = None
        self.rfm_data = None

    def load_data(self):
        """Load data from CSV file."""
        dtype = dict(InvoiceNo = np.object_,
                     StockCode = np.object_,
                     Description = np.object_,
                     Quantity = np.int64,
                     UnitPrice = np.float32,
                     CustomerID = np.object_,
                     Country = np.object_)
        
        self.df = pd.read_csv(self.data_path, 
                              dtype=dtype, 
                              parse_dates=['InvoiceDate'],
                              encoding='ISO-8859-1')
        
        # Change format of CustomerID to string
        self.df["CustomerID"] = (self.df["CustomerID"]
                                 .astype(str)
                                 .str.replace('.0', '', regex=False)
                                 .str.zfill(6))
        
        print("Dimensions of the dataset: ", self.df.shape)
        print(f"Records: {len(self.df):,}")

        return self.df
    
    def clean_data(self):
        """
        Clean the dataset by removing invalid records and focusing on UK customers.
        """
        # Add TotalPrice column
        self.df['TotalPrice'] = self.df['Quantity'] * self.df['UnitPrice']

        # Remove records with cancelled orders
        self.df = self.df[~self.df['InvoiceNo'].astype(str).str.startswith('C')]
        
        self.df_uk = self.df[self.df['Country'] == 'United Kingdom'].copy()

        self.df_uk = self.df_uk.dropna(subset=['CustomerID'])

        self.df_uk = self.df_uk[(self.df_uk['Quantity'] > 0) & (self.df_uk['UnitPrice'] > 0)]
        
        return self.df_uk
    
    def create_time_features(self):
        """
        Create time-based features for analysis.
        """
        self.df_uk['DayOfWeek'] = self.df_uk['InvoiceDate'].dt.day_name()
        self.df_uk['HourOfDay'] = self.df_uk['InvoiceDate'].dt.hour

    def calculate_rfm(self):
        """
        Calculate RFM (Recency, Frequency, Monetary) metrics for each customer.
        """
        snapshot_date = self.df_uk['InvoiceDate'].max() + dt.timedelta(days=1)

        self.rfm_data = self.df_uk.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (snapshot_date - x.max()).days, # Recency
            'InvoiceNo': lambda x: len(x.unique()), # Frequency
            'TotalPrice': lambda x: x.sum() # Monetary
        })

        self.rfm_data.columns = ['Recency', 'Frequency', 'Monetary']

        return self.rfm_data
    
    def save_cleaned_data(self, output_path = '../data/processed/'):
        """Save cleaned data to CSV file."""
        os.makedirs(output_path, exist_ok=True)
        self.df_uk.to_csv(os.path.join(output_path, 'cleaned_data_uk.csv'), index=False)
        print(f"Cleaned data saved to {output_path}")

class FeatureEngineer:
    """
    Create and transform features for clustering.
    """

    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.customer_features = None
        self.customer_features_transformed = None
        self.customer_features_scaled = None

        self.feature_customer = [
            "Sum_Quantity",
            "Mean_UnitPrice",
            "Mean_TotalPrice",
            "Sum_TotalPrice",
            "Count_Invoice",
            "Count_Stock",
            "Mean_InvoiceCountPerStock",
            "Mean_StockCountPerInvoice",
            "Mean_UnitPriceMeanPerInvoice",
            "Mean_QuantitySumPerInvoice",
            "Mean_TotalPriceMeanPerInvoice",
            "Mean_TotalPriceSumPerInvoice",
            "Mean_UnitPriceMeanPerStock",
            "Mean_QuantitySumPerStock",
            "Mean_TotalPriceMeanPerStock",
            "Mean_TotalPriceSumPerStock",
        ]      
        self.feature_customer2 = ["CustomerID"] + self.feature_customer
    
    def load_data(self):
        """
        Load cleaned data from CSV file.        
        """

        self.df = pd.read_csv(self.data_path)
        self.df['InvoiceDate'] = pd.to_datetime(self.df['InvoiceDate'])
        print("Dimensions of the dataset: ", self.df.shape)
        print(f"Records: {len(self.df):,}")
        return self.df
    
    def create_customer_features(self):
        """
        Create customer-level features for clustering.
        """
        num_customers = self.df['CustomerID'].nunique()
        self.customer_features = pd.DataFrame(
            data= np.zeros((num_customers, len(self.feature_customer2))),
            columns= self.feature_customer2
        )
        self.customer_features['CustomerID'] = self.customer_features['CustomerID'].astype('object')
        print('Calculating customer features...')
        for i, (customer_id, value) in enumerate(self.df.groupby('CustomerID')):
            # customer_id
            self.customer_features.iat[i, 0] = customer_id
            # Sum_Quantity
            self.customer_features.iat[i, 1] = value['Quantity'].sum()
            # Mean_UnitPrice
            self.customer_features.iat[i, 2] = value['UnitPrice'].mean()
            # Mean_TotalPrice
            self.customer_features.iat[i, 3] = value['TotalPrice'].mean()
            # Sum_TotalPrice
            self.customer_features.iat[i, 4] = value['TotalPrice'].sum()
            # Count_Invoice
            self.customer_features.iat[i, 5] = value['InvoiceNo'].nunique()
            # Count_Stock
            self.customer_features.iat[i, 6] = value['StockCode'].nunique()
            # 7-16. Another features
            self.customer_features.iat[i, 7] = value.groupby('StockCode').size().mean()  # Mean_InvoiceCountPerStock
            self.customer_features.iat[i, 8] = value.groupby('InvoiceNo').size().mean()  # Mean_StockCountPerInvoice
            self.customer_features.iat[i, 9] = value.groupby('InvoiceNo')['UnitPrice'].mean().mean()  # Mean_UnitPriceMeanPerInvoice
            self.customer_features.iat[i, 10] = value.groupby('InvoiceNo')['Quantity'].sum().mean()  # Mean_QuantitySumPerInvoice
            self.customer_features.iat[i, 11] = value.groupby('InvoiceNo')['TotalPrice'].mean().mean()  # Mean_TotalPriceMeanPerInvoice
            self.customer_features.iat[i, 12] = value.groupby('InvoiceNo')['TotalPrice'].sum().mean()  # Mean_TotalPriceSumPerInvoice
            self.customer_features.iat[i, 13] = value.groupby('StockCode')['UnitPrice'].mean().mean()  # Mean_UnitPriceMeanPerStock
            self.customer_features.iat[i, 14] = value.groupby('StockCode')['Quantity'].sum().mean()  # Mean_QuantitySumPerStock
            self.customer_features.iat[i, 15] = value.groupby('StockCode')['TotalPrice'].mean().mean()  # Mean_TotalPriceMeanPerStock
            self.customer_features.iat[i, 16] = value.groupby('StockCode')['TotalPrice'].sum().mean()  # Mean_TotalPriceSumPerStock
            if (i+1) % 500 == 0:
                print(f'Processed {i+1}/ {num_customers} customers...')
        print('Customer features calculation completed.')
        return self.customer_features

    def transform_features(self):
        """
        Apply Box-Cox transformation to customer features.
        """
        # Set CustomerID as index
        customer_features_indexed = self.customer_features.set_index('CustomerID')
        # Apply Box-Cox transformation
        feature_values = customer_features_indexed.values + 1

        self.customer_features_transformed = customer_features_indexed.copy()
        for i, feature in enumerate(self.feature_customer):
            transformed_data, _ = boxcox(feature_values[:, i])
            self.customer_features_transformed.iloc[:, i] = transformed_data

        print("Box-Cox transformation completed.")
        return self.customer_features_transformed
    
    def scale_features(self):
        """
        Scale customer features using StandardScaler.
        """
        scaler = StandardScaler()

        scaled_values = scaler.fit_transform(self.customer_features_transformed)

        self.customer_features_scaled = pd.DataFrame(
            data=scaled_values,
            columns=self.feature_customer,
            index=self.customer_features_transformed.index
        )

        print("Feature scaling completed.")
        joblib.dump(scaler, '../data/processed/scaler.pkl')
        return self.customer_features_scaled
    
    def plot_feature_boxplots(self, transformed: bool = False, save_path: str = None):
        """
        Plot boxplots for each customer feature.
        """
        if transformed and self.customer_features_transformed is not None:
            data_to_plot = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data_to_plot = self.customer_features.set_index('CustomerID')
            else:
                raise ValueError("Customer features not available for plotting.")
        
        with sns.plotting_context(context='notebook'):
            plt.figure(figsize=(15,15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i+1)
                sns.boxplot(data_to_plot.iloc[:,i] if transformed else data_to_plot[feature], color='skyblue')
                plt.title(f'Boxplot of {feature}', fontsize=10)
                plt.tight_layout() 

            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Boxplots saved to {save_path}")
            plt.show()

    def plot_feature_histograms(self, transformed: bool = False, save_path: str = None):
        """
        Plot histograms for each customer feature.
        """
        if transformed and self.customer_features_transformed is not None:
            data_to_plot = self.customer_features_transformed
        else:
            if self.customer_features is not None:
                data_to_plot = self.customer_features.set_index('CustomerID')
            else:
                raise ValueError("Customer features not available for plotting.")
        with sns.plotting_context(context='notebook'):
            plt.figure(figsize=(15,15))

            for i, feature in enumerate(self.feature_customer):
                plt.subplot(4, 4, i+1)
                sns.histplot(data_to_plot.iloc[:,i] if transformed else data_to_plot[feature], bins=30, color='skyblue', kde=True)
                plt.title(f'Histogram of {feature}', fontsize=10)
                plt.tight_layout() 

            if save_path:
                plt.savefig(save_path, dpi=300)
                print(f"Histograms saved to {save_path}")
            plt.show()

    def save_customer_features(self, output_path: str = '../data/processed/'):
        """Save customer features to CSV file."""
        os.makedirs(output_path, exist_ok=True)
        
        customer_features_indexed = self.customer_features.set_index('CustomerID')
        customer_features_indexed.to_csv(os.path.join(output_path, 'customer_features.csv'))

        self.customer_features_transformed.to_csv(os.path.join(output_path, 'customer_features_transformed.csv'))
        self.customer_features_scaled.to_csv(os.path.join(output_path, 'customer_features_scaled.csv'))
        print(f"Customer features saved to {output_path}")

class ClusteringModel:
    """
    Clustering model using KMeans.
    """

    FEATURE_NAMES = {
        'Sum_Quantity': "Sum of Quantity",
        'Mean_UnitPrice': "Mean of Unit Price",
        'Mean_TotalPrice': "Mean of Total Price",
        'Sum_TotalPrice': "Sum of Total Price",
        'Count_Invoice': "Count of Invoices",
        'Count_Stock': "Count of Stock Codes",
        'Mean_InvoiceCountPerStock': "Mean Invoice Count Per Stock",
        'Mean_StockCountPerInvoice': "Mean Stock Count Per Invoice",
        'Mean_UnitPriceMeanPerInvoice': "Mean of Unit Price Mean Per Invoice",
        'Mean_QuantitySumPerInvoice': "Mean of Quantity Sum Per Invoice",
        'Mean_TotalPriceMeanPerInvoice': "Mean of Total Price Mean Per Invoice",
        'Mean_TotalPriceSumPerInvoice': "Mean of Total Price Sum Per Invoice",
        'Mean_UnitPriceMeanPerStock': "Mean of Unit Price Mean Per Stock",
        'Mean_QuantitySumPerStock': "Mean of Quantity Sum Per Stock",
        'Mean_TotalPriceMeanPerStock': "Mean of Total Price Mean Per Stock",
        'Mean_TotalPriceSumPerStock': "Mean of Total Price Sum Per Stock",
    }

    def __init__(self, scaled_features_path: str, original_features_path: str):
        self.scaled_features_path = scaled_features_path
        self.original_features_path = original_features_path
        self.df_scaled = None
        self.df_original = None
        self.df_pca = None
        self.pca = None
        self.optimal_k = None
        self.cluster_results = None
        self.surrogate_models = {}
        self.shap_results = {}

    def load_data(self):
        """
        Load scaled and original customer features from CSV files.
        """
        self.df_scaled = pd.read_csv(self.scaled_features_path, index_col='CustomerID')
        self.df_original = pd.read_csv(self.original_features_path, index_col='CustomerID')
        print('Number of customers (scaled): ', self.df_scaled.shape[0])
        print('Feature dimensions (scaled): ', self.df_scaled.shape[1])
        return self.df_scaled, self.df_original
    
    def apply_pca(self, n_components: int = None):
        """
        Apply PCA to reduce feature dimensions.
        """
        self.pca = PCA(n_components=n_components)
        pca_result = self.pca.fit_transform(self.df_scaled)

        self.df_pca = pd.DataFrame(
            data=pca_result,
            columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
            index=self.df_scaled.index
        )
        print(f'PCA shape: {self.df_pca.shape}')
        return self.df_pca
    
    def plot_pca_variance(self, save_path: str = None):
        """
        Plot explained variance ratio of PCA components.
        """
        plt.figure(figsize=(12,6))

        plt.bar(
            range(1, len(self.pca.explained_variance_ratio_)+1),
            self.pca.explained_variance_ratio_,
            alpha=0.7,
            label='Individual explained variance'
        )

        plt.step(
            range(1, len(self.pca.explained_variance_ratio_)+1),
            np.cumsum(self.pca.explained_variance_ratio_),
            where='mid',
            label='Cumulative explained variance',
            color='red',
            linewidth=2
            )
        
        plt.axhline(y=0.8, color='orange', linestyle='--', label='80% Explained Variance')
        plt.axhline(y=0.9, color='green', linestyle='--', label='90% Explained Variance')
        plt.xlabel('Principal Component Index')
        plt.ylabel('Explained Variance Ratio')
        plt.title('PCA Explained Variance Ratio')
        plt.legend()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"PCA variance plot saved to {save_path}")
        plt.tight_layout()
        plt.show()
        print('\nCumulative Explained Variance Ratios:')
        for i in range(min(5, len(self.pca.explained_variance_ratio_))):
            cumsum = np.sum(self.pca.explained_variance_ratio_[:i+1])
            print(f'PC1 to PC{i+1}: {cumsum:.2%}')
    
    def determine_optimal_k(self, k_range=range(2,11)): 
        """
        Determine the optimal number of clusters using the silhouette score method.
        """
        inertias = []
        silhouette_scores = []

        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.df_pca)
            inertias.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(self.df_pca, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        self.optimal_k = {
            'k_range': k_range,
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'best_k_silhouette': list(k_range)[np.argmax(silhouette_scores)]
        }
        return self.optimal_k
    
    def plot_k_selection(self, save_path: str = None):
        """
        Plot Elbow method and Silhouette scores to select optimal k.
        """
        fig, axes = plt.subplots(1, 2, figsize=(16,5))

        # Elbow Method
        axes[0].plot(
            self.optimal_k['k_range'],
            self.optimal_k['inertias'],
            marker='o',
            linewidth=2,
            markersize=8,
            color='skyblue'
        )

        axes[0].set_title('Elbow Method for Optimal k', fontsize=14)
        axes[0].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[0].set_ylabel('Inertia', fontsize=12)
        axes[0].grid(True, alpha=0.3)

        # Silhouette Scores
        axes[1].plot(
            self.optimal_k['k_range'],
            self.optimal_k['silhouette_scores'],
            marker='o',
            linewidth=2,
            markersize=8,
            color='salmon'
        )

        axes[1].set_title('Silhouette Scores for Optimal k', fontsize=14)
        axes[1].set_xlabel('Number of clusters (k)', fontsize=12)
        axes[1].set_ylabel('Silhouette Score', fontsize=12)
        axes[1].grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"k selection plots saved to {save_path}")
        plt.tight_layout()
        plt.show()
        print(f"Optimal k by Silhouette Score: {self.optimal_k['best_k_silhouette']} (Score: {max(self.optimal_k['silhouette_scores']):.4f})")


    def fit_kmeans(self, k_values = [3,4]):
        """
        Fit KMeans clustering for specified k values and store results.
        """
        self.cluster_results = {}

        for k in k_values:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(self.df_pca)

            cluster_col = f'Cluster_k{k}'
            self.df_scaled[cluster_col] = cluster_labels
            self.df_pca[cluster_col] = cluster_labels
            self.df_original[cluster_col] = cluster_labels

            self.cluster_results[k] = {
                'labels': cluster_labels,
                'sizes': pd.Series(cluster_labels).value_counts().sort_index(),
                'means': self.df_original.groupby(cluster_col).mean(),
            }
            print(f'KMeans clustering completed for k={k}.')
        return self.cluster_results
    
    def plot_clusters_pca(self, k_values = [3,4], save_path: str = None):
        """
        Plot clusters in PCA-reduced space for specified k values.
        """
        fig, axes = plt.subplots(1, len(k_values), figsize=(6 * len(k_values), 5))
        if len(k_values) == 1:
            axes = [axes]

        for i, k in enumerate(k_values):
            cluster_col = f'Cluster_k{k}'
            scatter = axes[i].scatter(
                self.df_pca['PC1'],
                self.df_pca['PC2'],
                c=self.df_pca[cluster_col],
                cmap='viridis',
                alpha=0.6,
                s = 50
            )
            axes[i].set_title(f'KMeans Clusters (k={k}) in PCA Space', fontsize=14)
            axes[i].set_xlabel('Principal Component 1', fontsize=12)
            axes[i].set_ylabel('Principal Component 2', fontsize=12)
            plt.colorbar(scatter, ax=axes[i], label='Cluster Label')

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"Cluster PCA plots saved to {save_path}")
        plt.tight_layout()
        plt.show()

    def plot_clusters_pca_3d(self, k_values = [3,4], save_path: str = None):
        """
        Plot clusters in 3D PCA-reduced space for specified k values.
        """
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(16, 6))

        for i, k in enumerate(k_values):
            ax = fig.add_subplot(1, len(k_values), i+1, projection='3d')
            cluster_col = f'Cluster_k{k}'
            scatter = ax.scatter(
                self.df_pca['PC1'],
                self.df_pca['PC2'],
                self.df_pca['PC3'],
                c=self.df_pca[cluster_col],
                cmap='viridis',
                alpha=0.6,
                s=50
            )
            ax.set_title(f'KMeans Clusters (k={k}) in 3D PCA Space', fontsize=14)
            ax.set_xlabel('Principal Component 1', fontsize=12)
            ax.set_ylabel('Principal Component 2', fontsize=12)
            ax.set_zlabel('Principal Component 3', fontsize=12)
            plt.colorbar(scatter, ax=ax, label='Cluster Label')

        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"3D Cluster PCA plots saved to {save_path}")
        plt.tight_layout()
        plt.show()

    def create_radar_chart(self, k : int, cluster_names = None, save_path: str = None):
        """
        Create radar chart for cluster feature means.
        """
        cluster_means = self.cluster_results[k]['means']

        important_features = {
            "Sum_Quantity": "Khối lượng mua",
            "Sum_TotalPrice": "Tổng chi tiêu",
            "Mean_UnitPrice": "Mức giá ưa thích",
            "Count_Invoice": "Tần suất mua",
            "Count_Stock": "Đa dạng sản phẩm",
            "Mean_TotalPriceSumPerInvoice": "Giá trị/giao dịch",
        }

        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        data_normalized.columns = [important_features[col] for col in data_normalized.columns]

        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        colors = (
            ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
            if k == 4
            else ["#E74C3C", "#2ECC71", "#3498DB"]
        )
        if not cluster_names:
            cluster_names = [f'Cluster {i}' for i in range(k)]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        for i, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            values = row.tolist()
            values += values[:1]
            color = colors[i % len(colors)]
            cluster_name = cluster_names[i] if i < len(cluster_names) else f'Cluster {cluster_id}'
            ax.plot(
                angles,
                values,
                linewidth=4,
                linestyle='solid',
                label=cluster_name,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor='white',
                markeredgewidth=2
            )
            ax.fill(angles, values, alpha=0.15, color=color)

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, fontsize=12, weight='bold', color='#34495E')
            ax.set_ylim(0, 1)
            ax.set_yticklabels([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], size=10, color='#7F8C8D')
            ax.grid(True, alpha=0.3, color='#95A5A6', linewidth=0.8)
            ax.set_facecolor('#FAFAFA')
            ax.set_title(
            f"Phân tích phân khúc khách hàng (K={k})",
            size=16,
            weight="bold",
            pad=30,
            color="#2C3E50",
        )
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=12)

        plt.tight_layout()
        plt.show()

    def create_individual_radar_plots(self, k, cluster_names=None):
        """
        Tạo radar plot riêng cho từng cluster.

        Args:
            k (int): Number of clusters
            cluster_names (list): Custom names for clusters
        """
        cluster_means = self.cluster_results[k]["means"]

        # Chọn features quan trọng
        important_features = {
            "Sum_Quantity": "Khối lượng mua",
            "Sum_TotalPrice": "Tổng chi tiêu",
            "Mean_UnitPrice": "Mức giá ưa thích",
            "Count_Invoice": "Tần suất mua",
            "Count_Stock": "Đa dạng sản phẩm",
            "Mean_TotalPriceSumPerInvoice": "Giá trị/giao dịch",
            "Mean_TotalPriceMeanPerStock": "Chi tiêu/sản phẩm",
            "Mean_StockCountPerInvoice": "Sản phẩm/giao dịch",
        }

        feature_keys = list(important_features.keys())
        data_selected = cluster_means[feature_keys]

        # Chuẩn hóa dữ liệu
        global_min = data_selected.min()
        global_max = data_selected.max()
        data_normalized = (data_selected - global_min) / (global_max - global_min)
        data_normalized = data_normalized.fillna(0)

        # Thay thế labels tiếng Việt
        data_normalized.columns = [
            important_features[col] for col in data_normalized.columns
        ]

        # Setup angles
        categories = list(data_normalized.columns)
        N = len(categories)
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]

        # Colors chuyên nghiệp
        colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]
        if not cluster_names:
            cluster_names = [f"Nhóm {i}" for i in range(k)]

        # Tạo subplot cho từng cluster với layout tối ưu
        if k == 4:
            # Layout 2x2 cho k=4
            nrows, ncols = 2, 2
            figsize = (12, 10)
        else:
            # Layout 1 hàng cho các trường hợp khác
            nrows, ncols = 1, k
            figsize = (5 * k, 5)

        fig, axes = plt.subplots(
            nrows, ncols, figsize=figsize, subplot_kw=dict(projection="polar")
        )

        # Đảm bảo axes luôn là array 2D để dễ xử lý
        if k == 1:
            axes = np.array([[axes]])
        elif k == 4:
            # axes đã là 2D array (2x2)
            pass
        else:
            # Chuyển thành 2D array cho consistency
            axes = axes.reshape(1, -1)

        for idx, (cluster_id, row) in enumerate(data_normalized.iterrows()):
            # Tính toán vị trí trong grid 2D
            if k == 4:
                row_idx, col_idx = idx // 2, idx % 2
                ax = axes[row_idx, col_idx]
            else:
                ax = axes[0, idx] if len(axes.shape) == 2 else axes[idx]

            values = row.tolist()
            values += values[:1]

            color = colors[idx % len(colors)]
            cluster_name = (
                cluster_names[idx] if idx < len(cluster_names) else f"Cluster {idx}"
            )

            # Vẽ radar
            ax.plot(
                angles,
                values,
                "o-",
                linewidth=3,
                label=cluster_name,
                color=color,
                markersize=8,
                markerfacecolor=color,
                markeredgecolor="white",
                markeredgewidth=2,
            )
            ax.fill(angles, values, alpha=0.25, color=color)

            # Styling chuyên nghiệp
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories, size=11, weight="bold", color="#2C3E50")
            ax.set_ylim(0, 1)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(
                ["20%", "40%", "60%", "80%", "100%"], size=9, color="#7F8C8D"
            )
            ax.grid(True, alpha=0.3, color="#BDC3C7", linewidth=1)
            ax.set_facecolor("#FAFAFA")

            # Title cho mỗi subplot
            ax.set_title(
                f"{cluster_name}\n({cluster_means.index[idx]})",
                size=13,
                weight="bold",
                pad=20,
                color=color,
            )

        plt.suptitle(
            f"Phân tích chi tiết từng Cluster (K={k})", size=16, weight="bold", y=1.05
        )
        plt.tight_layout()
        plt.show()

    def train_surrogate_model(self, k):
        """
        Huấn luyện mô hình RandomForest classifier để có thể mô phỏng thuật toán KMeans.
        Mô hình này sẽ được dùng cho phân tích lời giải thích của SHAP.
        
        Args:
            k (int): Number of clusters
            
        Returns:
            dict: Training results including model and metrics
        """
        if k not in self.cluster_results:
            raise ValueError(f"Cluster results for k={k} not found. Run apply_kmeans first.")
        
        # Lấy tất cả cột không phải là Cluster_
        feature_cols = [col for col in self.df_scaled.columns if not col.startswith('Cluster_')]
        X = self.df_scaled[feature_cols].values
        y = self.cluster_results[k]['labels']
        
        # Huấn luyện mô hình RandomForest classifier
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf_model.fit(X, y)
        
        # Dự đoán
        y_pred = rf_model.predict(X)
        
        # Tính toán các chỉ số
        accuracy = accuracy_score(y, y_pred)
        conf_matrix = confusion_matrix(y, y_pred)
        if self.surrogate_models is None:
            self.surrogate_models = {}
        # Lưu kết quả
        self.surrogate_models[k] = {
            'model': rf_model,
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'feature_names': feature_cols
        }
        
        # In báo kết quả
        print(f"=== HUẤN LUYỆN MÔ HÌNH THAY THẾ (k={k}) ===")
        print(f"Độ chính xác: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nConfusion Matrix:")
        print(conf_matrix)
        print(f"\nMô hình có thể dự đoán {'CHÍNH XÁC' if accuracy >= 0.95 else 'HỢP LÝ'} các phân cụm.")
        
        return self.surrogate_models[k]
    
    def calculate_shap_values(self, k):
        """
        Tính toán SHAP values cho lời giải thích kết quả phân cụm sử dụng toàn bộ dữ liệu.
        
        Args:
            k (int): Number of clusters
            
        Returns:
            dict: SHAP explainer and values
        """
        if k not in self.surrogate_models:
            raise ValueError(f"Mô hình thay thế cho k={k} không tìm thấy. Vui lòng chạy train_surrogate_model trước.")
        
        # Lấy mô hình và các đặc trưng
        rf_model = self.surrogate_models[k]['model']
        feature_cols = self.surrogate_models[k]['feature_names']
        X = self.df_scaled[feature_cols].values
        
        # Tạo SHAP explainer với toàn bộ dữ liệu làm nền (background)
        # Khi dữ liệu làm nền càng lớn thì thuật toán SHAP càng chính xác
        print(f"Tính toán SHAP values cho {len(X):,} khách hàng...")
        explainer = shap.TreeExplainer(rf_model)
        shap_values_raw = explainer.shap_values(X)
        
        # Chuyển đổi sang định dạng list cho trường hợp đa lớp
        # Shape: (n_samples, n_features, n_classes) -> list (n_samples, n_features)
        if isinstance(shap_values_raw, np.ndarray) and len(shap_values_raw.shape) == 3:
            # TH đa lớp: chuyển vị để có (n_classes, n_samples, n_features)
            shap_values = [shap_values_raw[:, :, i] for i in range(shap_values_raw.shape[2])]
        else:
            # TH nhị phân: Đã ở định dạng list hoặc phân loại nhị phân
            shap_values = shap_values_raw
        if self.shap_results is None:
            self.shap_results = {}
        # Lưu kết quả
        self.shap_results[k] = {
            'explainer': explainer,
            'shap_values': shap_values,
            'feature_names': feature_cols,
            'X': X
        }
        
        print(f"Hoàn thành! SHAP values: {len(shap_values)} clusters, mỗi cluster shape: {shap_values[0].shape}")
        return self.shap_results[k]
    
    def plot_shap_summary(self, k, cluster_id=None):
        """
        Vẽ biểu đồ tóm tắt SHAP (beeswarm plot) cho phân tích cụm.
        
        Args:
            k (int): Number of clusters
            cluster_id (int, optional): Specific cluster to visualize. If None, shows all.
        """
        if k not in self.shap_results:
            raise ValueError(f"Giá trị SHAP cho k={k} không tìm thấy. Vui lòng chạy calculate_shap_values trước.")
        
        shap_values = self.shap_results[k]['shap_values']
        X = self.shap_results[k]['X']
        feature_names = self.shap_results[k]['feature_names']
    
        for i in range(k):
            shap.summary_plot(
                shap_values[i],
                X,
                feature_names=feature_names,
                max_display=3,
                show=True
            )

    def save_clusters(self, output_dir="../data/processed"):
        """
        Save cluster assignments.

        Args:
            output_dir (str): Output directory path
        """
        os.makedirs(output_dir, exist_ok=True)

        for k in self.cluster_results.keys():
            cluster_col = f"Cluster_{k}"
            self.df_original[cluster_col] = self.cluster_results[k]
            cluster_output = self.df_original[[cluster_col]].copy()
            cluster_output.columns = ["Cluster"]
            cluster_output = cluster_output.reset_index()
            cluster_output = cluster_output.sort_values(["Cluster", "CustomerID"])

            cluster_output.to_csv(
                f"{output_dir}/customer_clusters_k{k}.csv", index=False
            )
            print(
                f"Đã lưu kết quả phân cụm k={k}: {output_dir}/customer_clusters_k{k}.csv"
            )


class DataVisualizer:
    """
    A class for creating visualizations for customer segmentation analysis.

    This class provides methods for plotting various aspects of the data
    including temporal patterns, customer behavior, and cluster analysis.
    """

    def __init__(self):
        """Initialize the DataVisualizer with plotting settings."""
        plt.style.use("seaborn-v0_8-whitegrid")
        sns.set_palette("viridis")

    def plot_revenue_over_time(self, df):
        """
        Plot daily and monthly revenue patterns.

        Args:
            df (pd.DataFrame): Dataframe with InvoiceDate and TotalPrice columns
        """
        # Daily revenue
        plt.figure(figsize=(12, 5))
        daily_revenue = df.groupby(df["InvoiceDate"].dt.date)["TotalPrice"].sum()
        daily_revenue.plot()
        plt.title("Doanh thu hàng ngày")
        plt.xlabel("Ngày")
        plt.ylabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

        # Monthly revenue
        plt.figure(figsize=(12, 5))
        monthly_revenue = df.groupby(pd.Grouper(key="InvoiceDate", freq="M"))[
            "TotalPrice"
        ].sum()
        monthly_revenue.plot(kind="bar")
        plt.title("Doanh thu hàng tháng")
        plt.xlabel("Tháng")
        plt.ylabel("Doanh thu (GBP)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_time_patterns(self, df):
        """
        Plot purchase patterns by day and hour.

        Args:
            df (pd.DataFrame): Dataframe with time features
        """
        plt.figure(figsize=(12, 5))
        day_hour_counts = (
            df.groupby(["DayOfWeek", "HourOfDay"]).size().unstack(fill_value=0)
        )
        sns.heatmap(day_hour_counts, cmap="viridis")
        plt.title("Hoạt động mua hàng theo ngày và giờ")
        plt.xlabel("Giờ trong ngày")
        plt.ylabel("Ngày trong tuần (0=Thứ 2, 6=Chủ nhật)")
        plt.tight_layout()
        plt.show()

    def plot_product_analysis(self, df, top_n=10):
        """
        Plot top products by quantity and revenue.

        Args:
            df (pd.DataFrame): Transaction dataframe
            top_n (int): Number of top products to show
        """
        # Top sản phẩm theo số lượng
        plt.figure(figsize=(12, 5))
        top_products = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_products.values, y=top_products.index)
        plt.title(f"Top {top_n} sản phẩm theo số lượng bán")
        plt.xlabel("Số lượng bán")
        plt.tight_layout()
        plt.show()

        # Top sản phẩm theo doanh thu
        plt.figure(figsize=(12, 5))
        top_revenue_products = (
            df.groupby("Description")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(top_n)
        )
        sns.barplot(x=top_revenue_products.values, y=top_revenue_products.index)
        plt.title(f"Top {top_n} sản phẩm theo doanh thu")
        plt.xlabel("Doanh thu (GBP)")
        plt.tight_layout()
        plt.show()

    def plot_customer_distribution(self, df):
        """
        Plot customer behavior distributions.

        Args:
            df (pd.DataFrame): Transaction dataframe
        """
        # Số giao dịch trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        transactions_per_customer = df.groupby("CustomerID")["InvoiceNo"].nunique()
        sns.histplot(transactions_per_customer, bins=30, kde=True)
        plt.title("Phân phối số giao dịch trên mỗi khách hàng")
        plt.xlabel("Số giao dịch")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

        # Chi tiêu trên mỗi khách hàng
        plt.figure(figsize=(10, 5))
        spend_per_customer = df.groupby("CustomerID")["TotalPrice"].sum()
        spend_filter = spend_per_customer < spend_per_customer.quantile(0.99)
        sns.histplot(spend_per_customer[spend_filter], bins=30, kde=True)
        plt.title("Phân phối tổng chi tiêu trên mỗi khách hàng")
        plt.xlabel("Tổng chi tiêu (GBP)")
        plt.ylabel("Số khách hàng")
        plt.tight_layout()
        plt.show()

    def plot_rfm_analysis(self, rfm_data):
        """
        Plot RFM analysis visualizations.

        Args:
            rfm_data (pd.DataFrame): RFM dataframe
        """
        # RFM distributions
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))

        sns.histplot(rfm_data["Recency"], bins=30, kde=True, ax=axes[0])
        axes[0].set_title("Phân phối Recency (Ngày kể từ lần mua cuối)")
        axes[0].set_xlabel("Ngày")

        sns.histplot(rfm_data["Frequency"], bins=30, kde=True, ax=axes[1])
        axes[1].set_title("Phân phối Frequency (Số giao dịch)")
        axes[1].set_xlabel("Số giao dịch")

        monetary_filter = rfm_data["Monetary"] < rfm_data["Monetary"].quantile(0.99)
        sns.histplot(
            rfm_data.loc[monetary_filter, "Monetary"], bins=30, kde=True, ax=axes[2]
        )
        axes[2].set_title("Phân phối Monetary (Tổng chi tiêu)")
        axes[2].set_xlabel("Tổng chi tiêu (GBP)")

        plt.tight_layout()
        plt.show()