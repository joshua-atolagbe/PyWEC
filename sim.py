import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import matlab.engine
import scipy.io as sio
from tqdm import tqdm

class WECSimDataGenerator:
    """
    A class to generate training data for LSTM models using WEC-Sim.
    This interfaces with the MATLAB-based WEC-Sim to run simulations and collect data.
    """
    
    def __init__(self, wec_sim_path, wavestar_model_path):
        """
        Initialize the data generator.
        
        Args:
            wec_sim_path (str): Path to the WEC-Sim installation
            wavestar_model_path (str): Path to the Wavestar model for WEC-Sim
        """
        self.wec_sim_path = wec_sim_path
        self.wavestar_model_path = wavestar_model_path
        self.eng = None
        
    def start_matlab(self):
        """Start the MATLAB engine"""
        print("Starting MATLAB engine...")
        self.eng = matlab.engine.start_matlab()
        
        # Add WEC-Sim to MATLAB path
        self.eng.addpath(self.eng.genpath(self.wec_sim_path))
        self.eng.addpath(self.wavestar_model_path)
        print("MATLAB engine started and paths added.")
        
    def stop_matlab(self):
        """Stop the MATLAB engine"""
        if self.eng:
            self.eng.quit()
            self.eng = None
            print("MATLAB engine stopped.")
    
    def generate_wave_conditions(self, num_conditions=20):
        """
        Generate a range of wave conditions to simulate.
        
        Args:
            num_conditions (int): Number of different wave conditions to generate
            
        Returns:
            list: List of dictionaries with wave parameters
        """
        wave_conditions = []
        
        # Range of significant wave heights (Hs) and peak periods (Tp)
        # Based on typical values for Wavestar deployment
        hs_values = np.linspace(0.5, 3.0, num_conditions//4)  # Significant wave height (m)
        tp_values = np.linspace(4.0, 12.0, num_conditions//5)  # Peak period (s)
        
        # Create combinations of these parameters
        for hs in hs_values:
            for tp in tp_values:
                # JONSWAP is commonly used for North Sea conditions (where Wavestar was tested)
                wave_conditions.append({
                    'waveType': 'regular',  # Start with simpler regular waves
                    'height': hs,
                    'period': tp,
                })
                
                wave_conditions.append({
                    'waveType': 'irregular',
                    'spectrumType': 'JONSWAP',
                    'height': hs,  # Significant wave height for irregular waves
                    'period': tp,  # Peak period for irregular waves
                    'gamma': 3.3,  # JONSWAP peakedness parameter
                })
                
                if len(wave_conditions) >= num_conditions:
                    break
            if len(wave_conditions) >= num_conditions:
                break
                
        return wave_conditions[:num_conditions]  # Trim to requested number
    
    def run_simulation(self, wave_params, sim_time=300, dt=0.01, control_method='resistive'):
        """
        Run a WEC-Sim simulation with specified parameters.
        
        Args:
            wave_params (dict): Wave parameters
            sim_time (float): Simulation time in seconds
            dt (float): Time step in seconds
            control_method (str): Control method to use ('resistive', 'reactive', etc.)
            
        Returns:
            dict: Simulation results
        """
        if self.eng is None:
            self.start_matlab()
        
        # Create a temporary parameter file for this simulation
        param_script = f"""
        %% WEC-Sim Parameters for Wavestar simulation
        simu.simMechanicsFile = 'Wavestar.slx';
        simu.startTime = 0;
        simu.endTime = {sim_time};
        simu.dt = {dt};
        simu.rampTime = 50;
        
        %% Wave Parameters
        waves.type = '{wave_params['waveType']}';
        """
        
        if wave_params['waveType'] == 'regular':
            param_script += f"""
            waves.height = {wave_params['height']};
            waves.period = {wave_params['period']};
            """
        else:
            param_script += f"""
            waves.spectrumType = '{wave_params['spectrumType']}';
            waves.height = {wave_params['height']};
            waves.period = {wave_params['period']};
            waves.gamma = {wave_params['gamma']};
            """
        
        # Control parameters
        param_script += f"""
        %% Control Parameters
        controller.type = '{control_method}';
        """
        
        # Write the parameter script to a temp file
        temp_param_file = os.path.join(self.wavestar_model_path, 'temp_params.m')
        with open(temp_param_file, 'w') as f:
            f.write(param_script)
        
        # Run the simulation
        print(f"Running simulation for {wave_params['waveType']} waves with Hs={wave_params.get('height', 0)}, Tp={wave_params.get('period', 0)}")
        try:
            # Execute the parameter file
            self.eng.run(temp_param_file, nargout=0)
            
            # Run WEC-Sim
            self.eng.wecSim(nargout=0)
            
            # Load results from MATLAB workspace
            output_data = self.eng.workspace['output']
            
            # Convert MATLAB data to Python
            result = {
                'time': np.array(self.eng.eval('output.bodies.time')).flatten(),
                'angular_position': np.array(self.eng.eval('output.bodies(1).position')).flatten(),
                'angular_velocity': np.array(self.eng.eval('output.bodies(1).velocity')).flatten(),
                'excitation_moment': np.array(self.eng.eval('output.bodies(1).forceExcitation')).flatten(),
                'wave_elevation': np.array(self.eng.eval('output.waves.elevation')).flatten()
            }
            
            # Clean up
            os.remove(temp_param_file)
            
            return result
        
        except Exception as e:
            print(f"Simulation error: {e}")
            if os.path.exists(temp_param_file):
                os.remove(temp_param_file)
            return None
    
    def generate_dataset(self, output_dir='./wec_sim_data', num_conditions=10, sim_time=300):
        """
        Generate a complete dataset by running multiple simulations.
        
        Args:
            output_dir (str): Directory to save the data
            num_conditions (int): Number of different wave conditions to simulate
            sim_time (float): Simulation time for each condition
            
        Returns:
            pandas.DataFrame: Combined dataset from all simulations
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        wave_conditions = self.generate_wave_conditions(num_conditions)
        all_data = []
        
        for i, wave_params in enumerate(tqdm(wave_conditions, desc="Simulating wave conditions")):
            # Run simulation
            sim_result = self.run_simulation(wave_params, sim_time=sim_time)
            
            if sim_result is not None:
                # Create DataFrame for this simulation
                data = pd.DataFrame({
                    'time': sim_result['time'],
                    'angular_position': sim_result['angular_position'],
                    'angular_velocity': sim_result['angular_velocity'],
                    'excitation_moment': sim_result['excitation_moment'],
                    'wave_elevation': sim_result['wave_elevation'],
                    'wave_type': wave_params['waveType'],
                    'wave_height': wave_params.get('height', 0),
                    'wave_period': wave_params.get('period', 0)
                })
                
                # Save individual simulation data
                data.to_csv(f"{output_dir}/simulation_{i+1}.csv", index=False)
                
                # Add to combined dataset
                all_data.append(data)
                
                # Generate some plots
                self._plot_simulation_results(sim_result, f"{output_dir}/simulation_{i+1}_plot.png")
        
        # Combine and save all data
        if all_data:
            combined_data = pd.concat(all_data, ignore_index=True)
            combined_data.to_csv(f"{output_dir}/combined_dataset.csv", index=False)
            print(f"Dataset saved to {output_dir}/combined_dataset.csv")
            return combined_data
        else:
            print("No simulation data generated successfully.")
            return None
    
    def _plot_simulation_results(self, result, save_path):
        """Create plots of the simulation results"""
        fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
        
        # Plot wave elevation
        axs[0].plot(result['time'], result['wave_elevation'])
        axs[0].set_ylabel('Wave Elevation (m)')
        axs[0].grid(True)
        
        # Plot angular position
        axs[1].plot(result['time'], result['angular_position'])
        axs[1].set_ylabel('Angular Position (rad)')
        axs[1].grid(True)
        
        # Plot angular velocity
        axs[2].plot(result['time'], result['angular_velocity'])
        axs[2].set_ylabel('Angular Velocity (rad/s)')
        axs[2].grid(True)
        
        # Plot excitation moment
        axs[3].plot(result['time'], result['excitation_moment'])
        axs[3].set_xlabel('Time (s)')
        axs[3].set_ylabel('Excitation Moment (N·m)')
        axs[3].grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    def prepare_lstm_datasets(self, data_path, sequence_length=50, train_split=0.8):
        """
        Prepare datasets specifically formatted for LSTM training.
        
        Args:
            data_path (str): Path to the combined dataset CSV
            sequence_length (int): Length of input sequences for LSTM
            train_split (float): Proportion of data to use for training
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test) for both models
        """
        # Load the combined dataset
        data = pd.read_csv(data_path)
        
        # Prepare for Angular Velocity Prediction model (velocity based on time)
        sequences_velocity = []
        targets_velocity = []
        
        # Prepare for Excitation Moment Estimation model
        sequences_excitation = []
        targets_excitation = []
        
        # Iterate through each unique simulation (identified by wave parameters)
        grouped = data.groupby(['wave_type', 'wave_height', 'wave_period'])
        
        for _, group in grouped:
            # Sort by time
            group = group.sort_values('time')
            
            # Create sequences for velocity prediction
            for i in range(len(group) - sequence_length):
                # Input: Time and reference velocity (using current velocity as reference)
                seq_velocity = group.iloc[i:i+sequence_length][['time', 'angular_velocity']].values
                # Target: Next angular velocity
                target_velocity = group.iloc[i+sequence_length]['angular_velocity']
                
                sequences_velocity.append(seq_velocity)
                targets_velocity.append(target_velocity)
            
            # Create sequences for excitation moment estimation
            for i in range(len(group) - sequence_length):
                # Input: Angular position sequence
                seq_excitation = group.iloc[i:i+sequence_length][['angular_position', 'wave_elevation']].values
                # Target: Excitation moment
                target_excitation = group.iloc[i+sequence_length]['excitation_moment']
                
                sequences_excitation.append(seq_excitation)
                targets_excitation.append(target_excitation)
        
        # Convert to numpy arrays
        X_velocity = np.array(sequences_velocity)
        y_velocity = np.array(targets_velocity)
        X_excitation = np.array(sequences_excitation)
        y_excitation = np.array(targets_excitation)
        
        # Split into train and test sets
        train_size_velocity = int(len(X_velocity) * train_split)
        train_size_excitation = int(len(X_excitation) * train_split)
        
        # For velocity prediction
        X_train_velocity = X_velocity[:train_size_velocity]
        y_train_velocity = y_velocity[:train_size_velocity]
        X_test_velocity = X_velocity[train_size_velocity:]
        y_test_velocity = y_velocity[train_size_velocity:]
        
        # For excitation moment estimation
        X_train_excitation = X_excitation[:train_size_excitation]
        y_train_excitation = y_excitation[:train_size_excitation]
        X_test_excitation = X_excitation[train_size_excitation:]
        y_test_excitation = y_excitation[train_size_excitation:]
        
        # Save the prepared datasets
        np.save('X_train_velocity.npy', X_train_velocity)
        np.save('y_train_velocity.npy', y_train_velocity)
        np.save('X_test_velocity.npy', X_test_velocity)
        np.save('y_test_velocity.npy', y_test_velocity)
        
        np.save('X_train_excitation.npy', X_train_excitation)
        np.save('y_train_excitation.npy', y_train_excitation)
        np.save('X_test_excitation.npy', X_test_excitation)
        np.save('y_test_excitation.npy', y_test_excitation)
        
        return {
            'velocity_model': (X_train_velocity, y_train_velocity, X_test_velocity, y_test_velocity),
            'excitation_model': (X_train_excitation, y_train_excitation, X_test_excitation, y_test_excitation)
        }


# Example usage
if __name__ == '__main__':
    # Replace these paths with your actual WEC-Sim and model paths
    WEC_SIM_PATH = '/path/to/WEC-Sim'
    WAVESTAR_MODEL_PATH = '/path/to/WEC-Sim/examples/Wavestar'
    
    # Create data generator
    data_gen = WECSimDataGenerator(WEC_SIM_PATH, WAVESTAR_MODEL_PATH)
    
    try:
        # Generate dataset
        combined_data = data_gen.generate_dataset(
            output_dir='./wavestar_data',
            num_conditions=5,  # Start with a small number for testing
            sim_time=300       # 5 minutes of simulation data per condition
        )
        
        # Prepare LSTM-specific datasets
        if combined_data is not None:
            lstm_datasets = data_gen.prepare_lstm_datasets(
                data_path='./wavestar_data/combined_dataset.csv',
                sequence_length=50,  # 50 time steps as input sequence
                train_split=0.8
            )
            
            print("LSTM datasets prepared and saved.")
            
            # Show dataset shapes
            for model_name, (X_train, y_train, X_test, y_test) in lstm_datasets.items():
                print(f"\n{model_name} dataset:")
                print(f"X_train shape: {X_train.shape}")
                print(f"y_train shape: {y_train.shape}")
                print(f"X_test shape: {X_test.shape}")
                print(f"y_test shape: {y_test.shape}")
    
    finally:
        # Always stop MATLAB engine when done
        data_gen.stop_matlab()