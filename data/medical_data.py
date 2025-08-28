import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import medmnist
from medmnist import INFO
import matplotlib.pyplot as plt

class MedicalImageProcessor:
    """
    Enhanced medical image processor with domain-specific features
    Preserves spatial relationships and diagnostic relevance
    """
    
    def __init__(self, dataset_name='pathmnist', target_dim=8, random_seed=42):
        """
        Initialize the medical image processor
        
        Args:
            dataset_name: 'pathmnist' or 'pneumoniamnist'
            target_dim: Target dimensionality after preprocessing (must fit in quantum circuit)
            random_seed: For reproducibility
        """
        self.dataset_name = dataset_name.lower()
        self.target_dim = target_dim
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize scalers
        self.feature_scaler = StandardScaler()
        self.quantum_scaler = MinMaxScaler(feature_range=(0, 1))
        
        # Validate target_dim - can't be more than features we extract
        max_features = self._get_max_features()
        if target_dim > max_features:
            print(f"⚠️  Warning: target_dim={target_dim} exceeds max extractable features ({max_features})")
            print(f"    Setting target_dim to {max_features}")
            self.target_dim = max_features
        
        self.pca = PCA(n_components=self.target_dim, random_state=random_seed)
        self.is_fitted = False
        
    def load_dataset(self):
        """Load and return the specified medical dataset"""
        try:
            if self.dataset_name == 'pathmnist':
                DataClass = getattr(medmnist, 'PathMNIST')
                info = INFO['pathmnist']
            elif self.dataset_name == 'pneumoniamnist':
                DataClass = getattr(medmnist, 'PneumoniaMNIST') 
                info = INFO['pneumoniamnist']
            else:
                raise ValueError(f"Dataset {self.dataset_name} not supported")
            
            # Load train and test data
            train_dataset = DataClass(split='train', download=True)
            test_dataset = DataClass(split='test', download=True)
            val_dataset = DataClass(split='val', download=True)
            
            # Extract images and labels
            train_images, train_labels = train_dataset.imgs, train_dataset.labels.flatten()
            test_images, test_labels = test_dataset.imgs, test_dataset.labels.flatten()
            val_images, val_labels = val_dataset.imgs, val_dataset.labels.flatten()
            
            # Combine train and validation for cross-validation
            all_train_images = np.concatenate([train_images, val_images])
            all_train_labels = np.concatenate([train_labels, val_labels])
            
            print(f"Loaded {self.dataset_name}:")
            print(f"  Training samples: {len(all_train_images)}")
            print(f"  Test samples: {len(test_images)}")
            print(f"  Image shape: {train_images[0].shape}")
            print(f"  Number of classes: {len(np.unique(all_train_labels))}")
            
            return {
                'train_images': all_train_images,
                'train_labels': all_train_labels,
                'test_images': test_images,
                'test_labels': test_labels,
                'info': info
            }
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def extract_features(self, images):
        """
        Enhanced feature extraction preserving medical domain characteristics
        """
        features = []
        
        for img in images:
            # Flatten for basic statistics
            flat_img = img.flatten()
            
            # Core intensity statistics
            mean_intensity = np.mean(flat_img)
            std_intensity = np.std(flat_img)
            intensity_range = np.max(flat_img) - np.min(flat_img)
            
            # Intensity distribution features (medical relevance)
            q10, q25, q50, q75, q90 = np.percentile(flat_img, [10, 25, 50, 75, 90])
            
            # Skewness and kurtosis (shape of intensity distribution)
            skew = self._calculate_skewness(flat_img)
            kurt = self._calculate_kurtosis(flat_img)
            
            # Medical-specific features
            # High intensity regions (potential abnormalities)
            high_intensity_ratio = np.sum(flat_img > np.percentile(flat_img, 85)) / len(flat_img)
            
            # Low intensity regions (background/normal tissue)
            low_intensity_ratio = np.sum(flat_img < np.percentile(flat_img, 15)) / len(flat_img)
            
            # Contrast measure (important for medical imaging)
            if std_intensity > 0:
                contrast = std_intensity / mean_intensity
            else:
                contrast = 0.0
            
            # Spatial features (when possible)
            if len(img.shape) >= 2:
                # Edge detection approximation
                grad_x = np.mean(np.abs(np.gradient(img, axis=0)))
                grad_y = np.mean(np.abs(np.gradient(img, axis=1)))
                edge_strength = np.sqrt(grad_x**2 + grad_y**2)
                
                # Texture approximation via local variance
                kernel_size = min(3, img.shape[0]//4, img.shape[1]//4)
                if kernel_size > 0:
                    texture_variance = self._calculate_local_variance(img, kernel_size)
                else:
                    texture_variance = 0.0
            else:
                edge_strength = 0.0
                texture_variance = 0.0
            
            # Entropy (information content)
            entropy = self._calculate_entropy(flat_img)
            
            # Compile feature vector with medical relevance priority
            feature_vector = [
                mean_intensity,           # Overall brightness
                std_intensity,            # Intensity variation
                contrast,                 # Contrast ratio
                q50,                      # Median intensity
                intensity_range,          # Dynamic range
                high_intensity_ratio,     # Bright regions
                low_intensity_ratio,      # Dark regions
                edge_strength,            # Edge content
                texture_variance,         # Texture complexity
                entropy,                  # Information content
                skew,                     # Distribution asymmetry
                kurt,                     # Distribution peakedness
                q25,                      # Lower quartile
                q75,                      # Upper quartile
                grad_x if len(img.shape) >= 2 else 0.0  # Horizontal gradients
            ]
            
            features.append(feature_vector)
            
        return np.array(features)
    
    def _calculate_local_variance(self, img, kernel_size):
        """Calculate local variance as texture measure"""
        if kernel_size < 1:
            return 0.0
            
        variances = []
        h, w = img.shape[:2]
        
        for i in range(0, h - kernel_size + 1, kernel_size):
            for j in range(0, w - kernel_size + 1, kernel_size):
                patch = img[i:i+kernel_size, j:j+kernel_size]
                variances.append(np.var(patch))
        
        return np.mean(variances) if variances else 0.0
    
    def _get_max_features(self):
        """Get maximum number of features that can be extracted"""
        # Updated to reflect the enhanced feature extraction
        return 15
    
    def _calculate_entropy(self, data):
        """Calculate entropy of data"""
        # Simple entropy calculation
        hist, _ = np.histogram(data, bins=256, density=True)
        hist = hist[hist > 0]  # Remove zero probabilities
        return -np.sum(hist * np.log2(hist + 1e-10))
    
    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def fit_preprocessing(self, features):
        """Fit preprocessing transformations on training data"""
        print(f"Input features shape: {features.shape}")
        
        # Validate that we have enough features for the target dimension
        n_input_features = features.shape[1]
        if self.target_dim > n_input_features:
            print(f"⚠️  Adjusting target_dim from {self.target_dim} to {n_input_features}")
            self.target_dim = n_input_features
            self.pca = PCA(n_components=self.target_dim, random_state=self.random_seed)
        
        # Standardize features
        features_scaled = self.feature_scaler.fit_transform(features)
        
        # Apply PCA for dimensionality reduction (or keep same dimension)
        features_pca = self.pca.fit_transform(features_scaled)
        
        # Fit quantum-compatible scaling (0-1 range)
        self.quantum_scaler.fit(features_pca)
        
        self.is_fitted = True
        
        print(f"Preprocessing fitted:")
        print(f"  Original feature dimension: {features.shape[1]}")
        print(f"  Final dimension: {self.target_dim}")
        print(f"  Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.3f}")
        
        return self.transform_features(features)
    
    def transform_features(self, features):
        """Transform features using fitted preprocessing"""
        if not self.is_fitted:
            raise ValueError("Preprocessing not fitted. Call fit_preprocessing first.")
        
        # Apply same transformations
        features_scaled = self.feature_scaler.transform(features)
        features_pca = self.pca.transform(features_scaled)
        features_quantum = self.quantum_scaler.transform(features_pca)
        
        return features_quantum
    
    def prepare_for_quantum(self, dataset_dict):
        """
        Enhanced quantum preparation with medical feature preservation
        """
        # Extract features with medical domain specificity
        print("🏥 Extracting medical-specific features from training images...")
        train_features = self.extract_features(dataset_dict['train_images'])
        
        print("🏥 Extracting medical-specific features from test images...")
        test_features = self.extract_features(dataset_dict['test_images'])
        
        print(f"📊 Extracted {train_features.shape[1]} medical features per image")
        print("   Feature priorities: intensity stats, contrast, spatial gradients, texture")
        
        # Fit preprocessing on training data
        print("🔧 Fitting medical-aware preprocessing pipeline...")
        train_features_processed = self.fit_preprocessing(train_features)
        
        # Transform test data
        print("🔄 Transforming test data...")
        test_features_processed = self.transform_features(test_features)
        
        # Enhanced label processing for medical applications
        train_labels_processed = self._process_medical_labels(dataset_dict['train_labels'])
        test_labels_processed = self._process_medical_labels(dataset_dict['test_labels'])
        
        result = {
            'train_features': train_features_processed,
            'train_labels': train_labels_processed,
            'test_features': test_features_processed,
            'test_labels': test_labels_processed,
            'feature_dim': self.target_dim,
            'n_train': len(train_features_processed),
            'n_test': len(test_features_processed),
            'n_classes': len(np.unique(train_labels_processed)),
            'medical_feature_names': self._get_feature_names(),
            'class_distribution': {
                'train': dict(zip(*np.unique(train_labels_processed, return_counts=True))),
                'test': dict(zip(*np.unique(test_labels_processed, return_counts=True)))
            }
        }
        
        print(f"\n✅ Medical quantum-ready dataset prepared:")
        print(f"   Feature dimension: {result['feature_dim']}")
        print(f"   Training samples: {result['n_train']}")
        print(f"   Test samples: {result['n_test']}")
        print(f"   Classes: {result['n_classes']}")
        print(f"   Train distribution: {result['class_distribution']['train']}")
        print(f"   Test distribution: {result['class_distribution']['test']}")
        
        return result
    
    def _process_medical_labels(self, labels):
        """Process labels with medical application considerations"""
        # For medical applications, we might want to ensure balanced classes
        # or handle multi-class scenarios differently
        
        if self.dataset_name in ['pathmnist']:
            # PathMNIST has multiple pathology types - convert to binary for now
            # 0 = normal, 1+ = pathological
            return (labels > 0).astype(int)
        elif self.dataset_name in ['pneumoniamnist']:
            # PneumoniaMNIST is already binary (normal vs pneumonia)
            return labels.astype(int)
        else:
            # General binary conversion
            return (labels > 0).astype(int)
    
    def _get_feature_names(self):
        """Return descriptive names for extracted features"""
        return [
            'mean_intensity', 'std_intensity', 'contrast', 'median_intensity',
            'intensity_range', 'high_intensity_ratio', 'low_intensity_ratio',
            'edge_strength', 'texture_variance', 'entropy', 'skewness',
            'kurtosis', 'q25_intensity', 'q75_intensity', 'horizontal_gradient'
        ][:self.target_dim]
    
    def get_cross_validation_splits(self, features, labels, n_splits=5):
        """Generate stratified cross-validation splits"""
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_seed)
        return list(skf.split(features, labels))

# Usage example
if __name__ == "__main__":
    # Test the data pipeline with different dimensions
    for target_dim in [4, 6, 8, 10, 12, 15]:
        print(f"\n{'='*50}")
        print(f"Testing target_dim = {target_dim}")
        print(f"{'='*50}")
        
        processor = MedicalImageProcessor(dataset_name='pathmnist', target_dim=target_dim)
        
        # Load dataset
        dataset = processor.load_dataset()
        
        if dataset:
            # Prepare for quantum processing
            quantum_data = processor.prepare_for_quantum(dataset)
            
            print(f"✅ Successfully created {quantum_data['feature_dim']}D features")
        else:
            print(f"❌ Failed to load dataset")
            break