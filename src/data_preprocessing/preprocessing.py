import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple

def split_data(data: Dict[str, torch.Tensor], 
               train_ratio: float = 0.5, 
               test_ratio: float = 0.5,
               seed: int = 0) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Split data into train/test sets
    
    Args:
        data: Dictionary containing the data tensors
        train_ratio: Fraction for training set
        test_ratio: Fraction for test set
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_data, test_data)
    """
    # Validate ratios
    if abs(train_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + test_ratio must equal 1.0")
    
    # Get number of samples (assumes first dimension is batch)
    n_samples = data['states'].shape[0]
    
    # Calculate split sizes
    n_train = int(n_samples * train_ratio)
    n_test = n_samples - n_train
    
    # Create random indices
    torch.manual_seed(seed)
    indices = torch.randperm(n_samples)
    
    train_indices = indices[:n_train]
    test_indices = indices[n_train:]
    
    # Split the data
    train_data = {}
    test_data = {}
    
    for key, tensor in data.items():
        if key in ['states', 'parameter', 'parameters']:  # Keys that have sample dimension
            train_data[key] = tensor[train_indices]
            test_data[key] = tensor[test_indices]
        else:  # Keys like 't', 'x', 'y' that are shared across samples
            train_data[key] = tensor
            test_data[key] = tensor
    
    return train_data, test_data

def save_equation_data(train_data: Dict[str, torch.Tensor],
                      test_id_data: Dict[str, torch.Tensor],
                      test_od_data: Dict[str, torch.Tensor],
                      equation_folder: Path):
    """Save train/test_id/test_od data for an equation"""
    equation_folder = Path(equation_folder)
    equation_folder.mkdir(parents=True, exist_ok=True)
    
    torch.save(train_data, equation_folder / "data_train.pt")
    torch.save(test_id_data, equation_folder / "data_test_id.pt") 
    torch.save(test_od_data, equation_folder / "data_test_od.pt")
    
    print(f"Saved data to {equation_folder}/")

def save_equation_data_multiple_od(train_data: Dict[str, torch.Tensor],
                                  test_id_data: Dict[str, torch.Tensor],
                                  test_od_datasets: Dict[str, Dict[str, torch.Tensor]],
                                  equation_folder: Path):
    """Save train/test_id and multiple test_od datasets for an equation"""
    equation_folder = Path(equation_folder)
    equation_folder.mkdir(parents=True, exist_ok=True)
    
    torch.save(train_data, equation_folder / "data_train.pt")
    torch.save(test_id_data, equation_folder / "data_test_id.pt")
    
    for od_name, od_data in test_od_datasets.items():
        torch.save(od_data, equation_folder / f"data_test_{od_name}.pt")
    
    print(f"Saved data to {equation_folder}/ with OD sets: {list(test_od_datasets.keys())}")

def preprocess_advection_data(id_path: Path, od_path: Path, output_folder: Path):
    """Preprocess Advection equation data"""
    
    # Convert to Path objects if they're strings
    id_path = Path(id_path)
    od_path = Path(od_path)
    output_folder = Path(output_folder)
    
    # Process ID data
    states_id = np.load(id_path / "1D_Advection_Sols.npy")
    parameter_id = np.load(id_path / "beta_values.npy")
    t = np.load(id_path / "t_coordinate.npy")
    x = np.load(id_path / "x_coordinate.npy")
    
    id_data = {
         'states': torch.from_numpy(states_id).unsqueeze(-1),
         'parameter': torch.from_numpy(parameter_id).squeeze().unsqueeze(-1),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # Process OD data
    states_od = np.load(od_path / "1D_Advection_Sols.npy")
    parameter_od = np.load(od_path / "beta_values.npy")
    
    od_data = {
         'states': torch.from_numpy(states_od).unsqueeze(-1),
         'parameter': torch.from_numpy(parameter_od).squeeze().unsqueeze(-1),
         't': torch.from_numpy(t)[:-1],  # Same t and x coordinates
         'x': torch.from_numpy(x)
    }
    
    # Split ID data into train and test_id
    train_data, test_id_data = split_data(id_data)
    
    # Save all three datasets
    save_equation_data(train_data, test_id_data, od_data, output_folder)

def preprocess_burgers_data(id_path: Path, od_path: Path, output_folder: Path):
    """Preprocess Burgers equation data"""
    
    # Convert to Path objects if they're strings
    id_path = Path(id_path)
    od_path = Path(od_path)
    output_folder = Path(output_folder)
    
    # Process ID data
    states_id = np.load(id_path / "1D_Burgers_Sols.npy")
    parameter_id = np.load(id_path / "epsilon_values.npy")
    t = np.load(id_path / "t_coordinate.npy")
    x = np.load(id_path / "x_coordinate.npy")
    
    id_data = {
         'states': torch.from_numpy(states_id).unsqueeze(-1),
         'parameter': torch.from_numpy(parameter_id).squeeze().unsqueeze(-1),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # Process OD data
    states_od = np.load(od_path / "1D_Burgers_Sols.npy")
    parameter_od = np.load(od_path / "epsilon_values.npy")
    
    od_data = {
         'states': torch.from_numpy(states_od).unsqueeze(-1),
         'parameter': torch.from_numpy(parameter_od).squeeze().unsqueeze(-1),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # Split ID data into train and test_id
    train_data, test_id_data = split_data(id_data)
    
    # Save all three datasets
    save_equation_data(train_data, test_id_data, od_data, output_folder)

def preprocess_1d_diffusionreaction_data(id_path: Path, od_rho_path: Path, od_nu_path: Path, od_both_path: Path, output_folder: Path):
    """Preprocess 1D Diffusion-Reaction equation data"""
    
    # Convert to Path objects if they're strings
    id_path = Path(id_path)
    od_rho_path = Path(od_rho_path)
    od_nu_path = Path(od_nu_path)
    od_both_path = Path(od_both_path)
    output_folder = Path(output_folder)
    
    # Process ID data
    states_id = np.load(id_path / "ReacDiff_Sols.npy")
    parameter_1_id = np.load(id_path / "rho_values.npy")
    parameter_2_id = np.load(id_path / "nu_values.npy")
    t = np.load(id_path / "t_coordinate.npy")
    x = np.load(id_path / "x_coordinate.npy")
    
    id_data = {
         'states': torch.from_numpy(states_id).unsqueeze(-1),
         'parameter': torch.from_numpy(np.stack((parameter_1_id.squeeze(), parameter_2_id.squeeze()), axis=-1)),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # Process each OD dataset separately
    od_datasets = {}
    
    # OD rho data
    states_od_rho = np.load(od_rho_path / "ReacDiff_Sols.npy")
    parameter_1_od_rho = np.load(od_rho_path / "rho_values.npy")
    parameter_2_od_rho = np.load(od_rho_path / "nu_values.npy")
    od_datasets['od_rho'] = {
         'states': torch.from_numpy(states_od_rho).unsqueeze(-1),
         'parameter': torch.from_numpy(np.stack((parameter_1_od_rho.squeeze(), parameter_2_od_rho.squeeze()), axis=-1)),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # OD nu data
    states_od_nu = np.load(od_nu_path / "ReacDiff_Sols.npy")
    parameter_1_od_nu = np.load(od_nu_path / "rho_values.npy")
    parameter_2_od_nu = np.load(od_nu_path / "nu_values.npy")
    od_datasets['od_nu'] = {
         'states': torch.from_numpy(states_od_nu).unsqueeze(-1),
         'parameter': torch.from_numpy(np.stack((parameter_1_od_nu.squeeze(), parameter_2_od_nu.squeeze()), axis=-1)),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # OD both data
    states_od_both = np.load(od_both_path / "ReacDiff_Sols.npy")
    parameter_1_od_both = np.load(od_both_path / "rho_values.npy")
    parameter_2_od_both = np.load(od_both_path / "nu_values.npy")
    od_datasets['od_both'] = {
         'states': torch.from_numpy(states_od_both).unsqueeze(-1),
         'parameter': torch.from_numpy(np.stack((parameter_1_od_both.squeeze(), parameter_2_od_both.squeeze()), axis=-1)),
         't': torch.from_numpy(t)[:-1],
         'x': torch.from_numpy(x)
    }
    
    # Split ID data into train and test_id
    train_data, test_id_data = split_data(id_data)
    
    # Save all datasets
    save_equation_data_multiple_od(train_data, test_id_data, od_datasets, output_folder)

def preprocess_2d_gray_scott_data(train_path: Path, test_id_path: Path, test_od_path: Path, output_folder: Path):
    """Preprocess 2D Gray-Scott equation data (already split)"""
    
    # Convert to Path objects if they're strings
    train_path = Path(train_path)
    test_id_path = Path(test_id_path)
    test_od_path = Path(test_od_path)
    output_folder = Path(output_folder)
    
    # Load train data
    train_data_raw = np.load(train_path)
    train_data = {
        'states': torch.from_numpy(train_data_raw['u']).float(),
        'parameter': (torch.from_numpy(train_data_raw['beta']).float()-0.035)*100,
        'x': torch.from_numpy(train_data_raw['x']).float(),
        'y': torch.from_numpy(train_data_raw['y']).float(),
        't': torch.from_numpy(train_data_raw['t']).float()
    }
    
    # Load test_id data
    test_id_data_raw = np.load(test_id_path)
    test_id_data = {
        'states': torch.from_numpy(test_id_data_raw['u']).float(),
        'parameter': (torch.from_numpy(test_id_data_raw['beta']).float()-0.035)*100,
        'x': torch.from_numpy(test_id_data_raw['x']).float(),
        'y': torch.from_numpy(test_id_data_raw['y']).float(),
        't': torch.from_numpy(test_id_data_raw['t']).float()
    }
    
    # Load test_od data
    test_od_data_raw = np.load(test_od_path)
    test_od_data = {
        'states': torch.from_numpy(test_od_data_raw['u']).float(),
        'parameter': (torch.from_numpy(test_od_data_raw['beta']).float()-0.035)*100,
        'x': torch.from_numpy(test_od_data_raw['x']).float(),
        'y': torch.from_numpy(test_od_data_raw['y']).float(),
        't': torch.from_numpy(test_od_data_raw['t']).float()
    }

    
    # Save all three datasets
    save_equation_data(train_data, test_id_data, test_od_data, output_folder)

def preprocess_2d_navier_stokes_data(id_path: Path, od_path: Path, output_folder: Path):
    """Preprocess 2D Navier-Stokes equation data"""
    
    # Convert to Path objects if they're strings
    id_path = Path(id_path)
    od_path = Path(od_path)
    output_folder = Path(output_folder)
    
    def load_navier_stokes_data(path):
        state1 = np.load(path / "HD_Sols__Vx.npy")
        state2 = np.load(path / "HD_Sols__Vy.npy")
        state3 = np.load(path / "HD_Sols__P.npy")
        state4 = np.load(path / "HD_Sols__D.npy")
        parameter = np.load(path / "eta_values.npy")
        x = np.load(path / "x_coordinate.npy")
        y = np.load(path / "y_coordinate.npy")
        t = np.load(path / "t_coordinate.npy")
        
        states = np.stack((state1, state2, state3, state4), axis=-1).squeeze()

        states_t = torch.from_numpy(states)
        t = torch.from_numpy(t)
        states_t = states_t[:, ::2, ...]
        t_t =t[::2]
        
        return {
            'states': states_t,
            'parameter': torch.from_numpy(parameter).squeeze().unsqueeze(-1),
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            't': t_t
        }
    
    # Load ID and OD data
    id_data = load_navier_stokes_data(id_path)
    od_data = load_navier_stokes_data(od_path)
    
    # Split ID data into train and test_id
    train_data, test_id_data = split_data(id_data)
    
    # Save all three datasets
    save_equation_data(train_data, test_id_data, od_data, output_folder)

def preprocess_2d_diffusionreaction_data(train_path: Path, test_path: Path, output_folder: Path):
    """Preprocess 2D Diffusion-Reaction equation data"""
    
    # Convert to Path objects if they're strings
    train_path = Path(train_path)
    test_path = Path(test_path)
    output_folder = Path(output_folder)
    
    # Load train data
    train_data_raw = np.load(train_path)
    train_data = {
        'states': torch.from_numpy(train_data_raw['u'])[:,50::2,:,:,:].float(),
        'parameter': torch.from_numpy(train_data_raw['beta']).float(),
        'x': torch.from_numpy(train_data_raw['x']).float(),
        'y': torch.from_numpy(train_data_raw['y']).float(),
        't': torch.from_numpy(train_data_raw['t'])[50::2].float()
    }
    
    # Load test data (use as test_od)
    test_data_raw = np.load(test_path)
    test_od_data = {
        'states': torch.from_numpy(test_data_raw['u'])[:,50::2,:,:,:].float(),
        'parameter': torch.from_numpy(test_data_raw['beta']).float(),
        'x': torch.from_numpy(test_data_raw['x']).float(),
        'y': torch.from_numpy(test_data_raw['y']).float(),
        't': torch.from_numpy(test_data_raw['t'])[50::2].float()
    }
    
    # Split train data to get test_id
    train_final, test_id_data = split_data(train_data)
    
    # Save all three datasets
    save_equation_data(train_final, test_id_data, test_od_data, output_folder)

if __name__ == "__main__":
    
    # Advection
    preprocess_advection_data(
        "data/raw/Advection/Advection_id/", 
        "data/raw/Advection/Advection_od/", 
        "data/processed/Advection/"
    )
    
    # Advection small steps
    preprocess_advection_data(
        "data/raw/Advection/Advection_id_smallsteps/", 
        "data/raw/Advection/Advection_od_smallsteps/", 
        "data/processed/Advection_smallsteps/"
    )
    
    # Burgers
    preprocess_burgers_data(
        "data/raw/Burgers/Burgers_id/", 
        "data/raw/Burgers/Burgers_od/", 
        "data/processed/Burgers/"
    )
    
    # Burgers small steps
    preprocess_burgers_data(
        "data/raw/Burgers/Burgers_id_smallsteps/", 
        "data/raw/Burgers/Burgers_od_smallsteps/", 
        "data/processed/Burgers_smallsteps/"
    )
    
    # Reaction Diffusion 1D
    preprocess_1d_diffusionreaction_data(
        "data/raw/ReactionDiffusion_1D/ReacDiff_id/",
        "data/raw/ReactionDiffusion_1D/ReacDiff_od_rho/",
        "data/raw/ReactionDiffusion_1D/ReacDiff_od_nu/",
        "data/raw/ReactionDiffusion_1D/ReacDiff_od_both/",
        "data/processed/ReactionDiffusion_1D/"
    )
    
    # Gray-Scott
    preprocess_2d_gray_scott_data(
        "data/raw/GrayScott/gray_scott_train.npz",
        "data/raw/GrayScott/gray_scott_test_id.npz",
        "data/raw/GrayScott/gray_scott_test_od.npz",
        "data/processed/GrayScott/"
    )
    
    # Navier-Stokes
    preprocess_2d_navier_stokes_data(
        "data/raw/NavierStokes/CFD_train/",
        "data/raw/NavierStokes/CFD_od/",
        "data/processed/NavierStokes/"
    )
    
    # Reaction Diffusion 2D
    preprocess_2d_diffusionreaction_data(
        "data/raw/ReactionDiffusion_2D/2D_reacdiff_train.npz",
        "data/raw/ReactionDiffusion_2D/2D_reacdiff_test.npz",
        "data/processed/ReactionDiffusion_2D/"
    )