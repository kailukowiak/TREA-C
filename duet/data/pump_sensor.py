"""Pump Sensor Dataset - Industrial pump vibration data.

This uses the publicly available pump sensor dataset from Kaggle,
but we'll create a synthetic version that mimics its characteristics.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset


class PumpSensorDataset(Dataset):
    """Industrial pump sensor dataset for fault classification.
    
    This dataset simulates industrial pump sensors with:
    - Vibration sensors (3-axis accelerometer)
    - Temperature sensors
    - Pressure sensors
    - Flow rate sensors
    - Categorical features: pump type, location, operational mode
    """
    
    def __init__(
        self,
        data_dir: str = "./data/pump_sensor",
        train: bool = True,
        sequence_length: int = 100,
        num_samples: int = 5000,
        task: str = "classification",
        num_classes: int = 4,  # Normal, Cavitation, Imbalance, Bearing Fault
        create_data: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.train = train
        self.sequence_length = sequence_length
        self.task = task
        self.num_classes = num_classes
        
        if create_data:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._create_synthetic_pump_data(num_samples)
        
        # Load data
        self._load_data()
    
    def _create_synthetic_pump_data(self, num_samples):
        """Create synthetic pump sensor data."""
        np.random.seed(42 if self.train else 123)
        
        # Fault types
        fault_names = ["Normal", "Cavitation", "Imbalance", "Bearing_Fault"]
        
        all_sequences = []
        all_labels = []
        
        for sample_idx in range(num_samples):
            # Randomly assign fault type
            fault_type = np.random.choice(self.num_classes)
            
            # Generate base signals based on fault type
            t = np.linspace(0, 10, self.sequence_length)
            
            # Vibration signals (3-axis)
            if fault_type == 0:  # Normal
                vib_x = 0.1 * np.random.randn(self.sequence_length) + 0.05 * np.sin(2 * np.pi * 1 * t)
                vib_y = 0.1 * np.random.randn(self.sequence_length) + 0.05 * np.cos(2 * np.pi * 1 * t)
                vib_z = 0.1 * np.random.randn(self.sequence_length)
                temp = 70 + 5 * np.random.randn() + 0.1 * t  # Slight warming
                pressure = 100 + 2 * np.random.randn(self.sequence_length)
                flow = 50 + 1 * np.random.randn(self.sequence_length)
                
            elif fault_type == 1:  # Cavitation
                # Cavitation causes erratic vibration and pressure drops
                vib_x = 0.3 * np.random.randn(self.sequence_length) + 0.2 * np.sin(2 * np.pi * 5 * t)
                vib_y = 0.3 * np.random.randn(self.sequence_length) + 0.2 * np.cos(2 * np.pi * 5 * t)
                vib_z = 0.4 * np.random.randn(self.sequence_length)
                temp = 75 + 8 * np.random.randn() + 0.2 * t
                pressure = 90 + 10 * np.random.randn(self.sequence_length)  # Lower, erratic
                flow = 45 + 5 * np.random.randn(self.sequence_length)  # Reduced flow
                
            elif fault_type == 2:  # Imbalance
                # Imbalance causes periodic vibration at rotation frequency
                rotation_freq = 2.5
                vib_x = 0.15 * np.random.randn(self.sequence_length) + 0.4 * np.sin(2 * np.pi * rotation_freq * t)
                vib_y = 0.15 * np.random.randn(self.sequence_length) + 0.4 * np.cos(2 * np.pi * rotation_freq * t)
                vib_z = 0.1 * np.random.randn(self.sequence_length) + 0.1 * np.sin(2 * np.pi * rotation_freq * t)
                temp = 72 + 6 * np.random.randn() + 0.15 * t
                pressure = 98 + 3 * np.random.randn(self.sequence_length)
                flow = 48 + 2 * np.random.randn(self.sequence_length)
                
            else:  # Bearing Fault
                # Bearing faults cause high-frequency vibration with impacts
                vib_x = 0.2 * np.random.randn(self.sequence_length) + 0.1 * np.sin(2 * np.pi * 8 * t)
                vib_y = 0.2 * np.random.randn(self.sequence_length) + 0.1 * np.cos(2 * np.pi * 8 * t)
                vib_z = 0.25 * np.random.randn(self.sequence_length)
                # Add random impacts
                for _ in range(np.random.randint(5, 15)):
                    impact_loc = np.random.randint(0, self.sequence_length)
                    vib_x[max(0, impact_loc-2):min(self.sequence_length, impact_loc+2)] += np.random.uniform(0.5, 1.5)
                temp = 78 + 10 * np.random.randn() + 0.3 * t  # Higher temp
                pressure = 102 + 4 * np.random.randn(self.sequence_length)
                flow = 49 + 2 * np.random.randn(self.sequence_length)
            
            # Stack features
            numeric_features = np.stack([
                vib_x, vib_y, vib_z, 
                np.full(self.sequence_length, temp) if isinstance(temp, (int, float)) else temp,
                pressure, flow
            ])
            
            # Add some missing values
            mask = np.random.random(numeric_features.shape) < 0.05
            numeric_features[mask] = np.nan
            
            all_sequences.append(numeric_features)
            all_labels.append(fault_type)
        
        # Convert to tensors
        self.sequences = torch.FloatTensor(np.array(all_sequences))
        self.labels = torch.LongTensor(all_labels)
        
        # Create categorical features (pump type, location, mode)
        # These don't change within a sequence
        self.cat_features = torch.zeros(num_samples, 3, self.sequence_length, dtype=torch.long)
        for i in range(num_samples):
            self.cat_features[i, 0, :] = torch.randint(0, 3, (1,))  # Pump type (0-2)
            self.cat_features[i, 1, :] = torch.randint(0, 5, (1,))  # Location (0-4)
            self.cat_features[i, 2, :] = torch.randint(0, 2, (1,))  # Mode (0-1)
    
    def _load_data(self):
        """Load data (already created in memory)."""
        pass
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'x_num': self.sequences[idx],  # [6, T]
            'x_cat': self.cat_features[idx],  # [3, T]
            'y': self.labels[idx]
        }
    
    def get_feature_info(self):
        """Get information about features."""
        return {
            'n_numeric': 6,  # vib_x, vib_y, vib_z, temp, pressure, flow
            'n_categorical': 3,  # pump_type, location, mode
            'cat_cardinalities': [3, 5, 2],
            'numeric_names': ['vib_x', 'vib_y', 'vib_z', 'temperature', 'pressure', 'flow_rate'],
            'categorical_names': ['pump_type', 'location', 'operational_mode'],
            'fault_types': ["Normal", "Cavitation", "Imbalance", "Bearing_Fault"]
        }