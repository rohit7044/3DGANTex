"""
Configuration file for 3D face reconstruction and editing pipeline.
Contains all paths and parameters used across the application.
"""

from pathlib import Path
import yaml

class Config:
    def __init__(self,input_dir, output_dir):
        # Directory paths
        self.current_directory = Path.cwd()
        self.input_dir = input_dir
        self.output_dir = output_dir

        # Model paths
        self.psp_model_path = self.current_directory / "pretrained_models/restyle_pSp_ffhq.pt"
        self.e4e_model_path = self.current_directory / "pretrained_models/restyle_e4e_ffhq.pt"
        self.shape_predictor_path = self.current_directory / "pretrained_models/shape_predictor_68_face_landmarks.dat"
        self.threeddfa_configs_path = self.current_directory / "ThreeDDFA_configs/mb1_120x120.yml"
        
        # Load 3DDFA configs
        self.threeddfa_config = self._load_threeddfa_config()
        
    def _load_threeddfa_config(self):
        """Load and process 3DDFA configuration file."""
        try:
            with open(self.threeddfa_configs_path) as f:
                cfg = yaml.safe_load(f)
                return {
                    **cfg,
                    'checkpoint_fp': str(self.current_directory / cfg['checkpoint_fp']),
                    'bfm_fp': str(self.current_directory / cfg['bfm_fp'])
                }
        except Exception as e:
            raise RuntimeError(f"Failed to load 3DDFA config: {str(e)}")

    def get_output_paths(self):
        """Generate output file paths based on input image path."""
        # Use the input_path to get the base name
        base_name = self.input_dir.split("/")[-1].replace(".png", "")
        
        # Define the output directory and ensure it exists
        output_dir = Path(self.output_dir)  # Convert to Path object
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the directory if it does not exist

        return {
            'pose': output_dir / f"{base_name}_pose.png",
            'uv_tex': output_dir / f"{base_name}_uv_tex.png",
            'obj': output_dir / f"{base_name}_obj.obj"
        }