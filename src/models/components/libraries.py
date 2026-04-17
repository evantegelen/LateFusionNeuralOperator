"""
Function libraries for sparse regression in PDE modeling.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
import itertools
from math import comb

class BaseLibrary(nn.Module, ABC):
    """Base class for function libraries"""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, hidden_states: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Build library from hidden states and parameters
        
        Args:
            hidden_states: [B, X, H] - Hidden states from FNO
            parameters: [B, P] - Input parameters    
        Returns:
            library: [B, X, L] - Library functions where L is library size
        """
        pass
    
    @abstractmethod
    def get_library_size(self, hidden_dim: int, param_dim: int) -> int:
        """Return the size of the library given input dimensions"""
        pass
    
    @abstractmethod
    def get_feature_names(self, hidden_dim: int, param_dim: int) -> List[str]:
        """Return list of feature names for interpretation"""
        pass


class IdentityLibrary(BaseLibrary):
    """Identity library: maps all inputs to themselves"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, hidden_states: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, X, H]
            parameters: [B, P]
        Returns:
            library: [B, X, H+P] - Concatenated hidden states and parameters
        """
        batch_size, spatial_size, hidden_dim = hidden_states.shape
        param_dim = parameters.shape[1]
        
        parameters_expanded = parameters.unsqueeze(1).expand(-1, spatial_size, -1) #[B, X, P] Expand parameters to spatial dimensions

        library = torch.cat([hidden_states, parameters_expanded], dim=-1)  # [B, X, H+P] Concatenate hidden states and parameters        
        return library
    
    def get_library_size(self, hidden_dim: int, param_dim: int) -> int:
        return hidden_dim + param_dim
    
    def get_feature_names(self, hidden_dim: int, param_dim: int) -> List[str]:
        """Return feature names for identity library"""
        feature_names = []
        
        # Hidden state names: u1, u2, u3, ...
        for i in range(hidden_dim):
            feature_names.append(f"u{i+1}")
        
        # Parameter names: p1, p2, p3, ...
        for i in range(param_dim):
            feature_names.append(f"p{i+1}")
        
        return feature_names

class PolynomialLibrary(BaseLibrary):
    """Polynomial library: creates polynomial combinations of features"""
    
    def __init__(self, order_states: int = 2, order_parameters: int = 1, include_interactions: bool = True, num_variables: int = 1, num_parameters: int = 1, hidden_dim: int = 4):
        super().__init__()
        self.order_states = order_states
        self.order_parameters = order_parameters
        self.include_interactions = include_interactions
        self.num_parameters = num_parameters
        self.num_variables = num_variables
        self.hidden_dim = hidden_dim
        self.library_size = self.get_library_size()
        
    def forward(self, hidden_states: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, X, H]
            parameters: [B, P]
        Returns:
            library: [B, X, L] - Polynomial library
        """
        batch_size, spatial_size, hidden_dim = hidden_states.shape
        parameters_expanded = parameters.unsqueeze(1).expand(-1, spatial_size, -1)  # [B, X, P]
        
        # Step 1: Create polynomial library for states (including bias)
        states_library = self._create_polynomial_terms(hidden_states, self.order_states)  # [B, X, L_states]
        # Step 2: Create polynomial library for parameters (including bias)
        params_library = self._create_polynomial_terms(parameters_expanded, self.order_parameters)  # [B, X, L_params]
        
        # Step 3: Combine libraries based on interaction setting
        if self.include_interactions:
            combined_library = self._combine_libraries(states_library, params_library)  # [B, X, L_states * L_params]
        else:
            combined_library = torch.cat([states_library, params_library], dim=-1)  # [B, X, L_states + L_params]
        
        return combined_library

    def _create_polynomial_terms(self, features: torch.Tensor, max_order: int) -> torch.Tensor:
        """
        Create all polynomial terms up to max_order for given features (always includes bias)
        
        Args:
            features: [B, X, F] - Input features
            max_order: Maximum polynomial order
            
        Returns:
            polynomial_terms: [B, X, L] - All polynomial combinations including bias
        """
        batch_size, spatial_size, n_features = features.shape
        all_terms = []
        
        # Always add bias term (degree 0)
        bias_term = torch.ones(batch_size, spatial_size, 1, device=features.device)
        all_terms.append(bias_term)
        
        # Generate polynomial terms for each degree from 1 to max_order
        for degree in range(1, max_order + 1):
            degree_terms = self._generate_degree_terms(features, degree)
            all_terms.extend(degree_terms)
        
        # Concatenate all terms
        return torch.cat(all_terms, dim=-1)  # [B, X, sum of all combinations]
    
    def _generate_degree_terms(self, features: torch.Tensor, degree: int) -> List[torch.Tensor]:
        """Generate all polynomial terms of a specific degree"""
        batch_size, spatial_size, n_features = features.shape
        terms = []
        
        # Generate all combinations with replacement for the given degree
        for combination in itertools.combinations_with_replacement(range(n_features), degree):
            term = torch.ones(batch_size, spatial_size, 1, device=features.device)
            for feature_idx in combination:
                term = term * features[:, :, feature_idx:feature_idx+1]
            terms.append(term)
        return terms
    
    def _combine_libraries(self, states_library: torch.Tensor, params_library: torch.Tensor) -> torch.Tensor:
        """
        Combine states and parameters libraries by taking all products (interactions)
        
        Args:
            states_library: [B, X, L_states]
            params_library: [B, X, L_params]
            
        Returns:
            combined: [B, X, L_states * L_params]
        """
        n_states_terms = states_library.shape[2]
        n_params_terms = params_library.shape[2]
        
        combined_terms = []
        
        # Take all combinations of state terms with parameter terms
        for i in range(n_states_terms):
            for j in range(n_params_terms):
                # Multiply state term with parameter term
                combined_term = states_library[:, :, i:i+1] * params_library[:, :, j:j+1]  # [B, X, 1]
                combined_terms.append(combined_term)
        
        return torch.cat(combined_terms, dim=-1)  # [B, X, L_states * L_params]
    
    def get_library_size(self) -> int:
        """Calculate total library size (always includes bias terms)"""
        
        # Calculate size of states polynomial library (including bias)
        states_size = 1  # bias term
        for degree in range(1, self.order_states + 1):
            states_size += comb(self.hidden_dim + degree - 1, degree)
        
        # Calculate size of parameters polynomial library (including bias)
        params_size = 1  # bias term
        for degree in range(1, self.order_parameters + 1):
            params_size += comb(self.num_parameters + degree - 1, degree)
        
        if self.include_interactions:
            # Combined size is product of both libraries (all interactions)
            combined_size = states_size * params_size
        else:
            # Combined size is sum of both libraries (no interactions)
            combined_size = states_size + params_size
            
        return combined_size
    
    def get_feature_names(self, hidden_dim: int, param_dim: int) -> List[str]:
        """Return feature names for polynomial library"""
        
        # Create state feature names: u1, u2, u3, ...
        state_names = [f"u{i+1}" for i in range(hidden_dim)]
        
        # Create parameter feature names: p1, p2, p3, ...
        param_names = [f"p{i+1}" for i in range(param_dim)]
        
        # Generate states polynomial feature names
        states_features = self._generate_polynomial_feature_names(state_names, self.order_states)
        
        # Generate parameters polynomial feature names
        params_features = self._generate_polynomial_feature_names(param_names, self.order_parameters)
        
        if self.include_interactions:
            # Create all combinations of state and parameter features
            combined_features = []
            for state_feature in states_features:
                for param_feature in params_features:
                    if state_feature == "1" and param_feature == "1":
                        combined_features.append("1")
                    elif state_feature == "1":
                        combined_features.append(param_feature)
                    elif param_feature == "1":
                        combined_features.append(state_feature)
                    else:
                        combined_features.append(f"{state_feature}*{param_feature}")
            return combined_features
        else:
            # Simply concatenate state and parameter features
            return states_features + params_features
    
    def _generate_polynomial_feature_names(self, feature_names: List[str], max_order: int) -> List[str]:
        """Generate polynomial feature names for given base features"""
        all_features = []
        
        # Always add bias term
        all_features.append("1")
        
        # Generate polynomial terms for each degree
        for degree in range(1, max_order + 1):
            degree_features = self._generate_degree_feature_names(feature_names, degree)
            all_features.extend(degree_features)
        
        return all_features
    
    def _generate_degree_feature_names(self, feature_names: List[str], degree: int) -> List[str]:
        """Generate feature names for a specific polynomial degree"""
        feature_strings = []
        n_features = len(feature_names)
        
        # Generate all combinations with replacement for the given degree
        for combination in itertools.combinations_with_replacement(range(n_features), degree):
            # Create feature name string
            if degree == 1:
                feature_strings.append(feature_names[combination[0]])
            else:
                # For higher degrees, create products like "u1*u2" or "u1^2"
                feature_counts = {}
                for idx in combination:
                    feature_counts[idx] = feature_counts.get(idx, 0) + 1
                
                feature_parts = []
                for idx, count in sorted(feature_counts.items()):
                    if count == 1:
                        feature_parts.append(feature_names[idx])
                    else:
                        feature_parts.append(f"{feature_names[idx]}^{count}")
                
                feature_strings.append("*".join(feature_parts))
        
        return feature_strings
    

class SimpleLibrary(BaseLibrary):
    """Simple library that just returns the parameter multiplied with the hidden states so:
    library = [p1*h1, p2*h2, h3, h4] if hidden_dim=4 and n_parameters = 2 where p1 is the first parameter and u1, u2, ... are the hidden states
    """
    
    def __init__(self, num_variables: int = 1, num_parameters: int = 1, hidden_dim: int = 4):
        super().__init__()
        self.num_variables = num_variables
        self.num_parameters = num_parameters
        self.hidden_dim = hidden_dim
        self.library_size = self.get_library_size()

        if self.num_parameters> self.hidden_dim:
            raise ValueError(f"Number of parameters ({self.num_parameters}) cannot be greater than hidden dimension ({self.hidden_dim})")
                
    def forward(self, hidden_states: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:
        batch_size, spatial_size, hidden_dim = hidden_states.shape
        parameters_expanded = parameters.unsqueeze(1).expand(-1, spatial_size, -1)  # [B, X, P]
        
        library = self._create_library(hidden_states, parameters_expanded)
        return library
    
    def _create_library(self, hidden_states: torch.Tensor, parameters: torch.Tensor) -> torch.Tensor:

        batch_size, spatial_size, hidden_dim = hidden_states.shape
        param_dim = parameters.shape[-1]

        # Create library by multiplying each hidden state with the first parameter and concatenating the remaining hidden states
        param_multiplied = hidden_states[:, :, :param_dim] * parameters  # [B, X, P]
        remaining_hidden = hidden_states[:, :, param_dim:]  # [B, X, H-P]
                
        library = torch.cat([param_multiplied, remaining_hidden], dim=-1)  # [B, X, H]
        return library
    
    def get_library_size(self) -> int:
        return self.hidden_dim
    
    def get_feature_names(self, hidden_dim: int, param_dim: int) -> List[str]:
        feature_names = []
        # Parameter-multiplied hidden state names: p1*u1, p2*u2, ...
        for i in range(param_dim):
            feature_names.append(f"p{i+1}*u{i+1}")
        # Remaining hidden state names: u{param_dim+1}, u{param_dim+2}, ...
        for i in range(param_dim, hidden_dim):
            feature_names.append(f"u{i+1}")
        return feature_names
    
    
    