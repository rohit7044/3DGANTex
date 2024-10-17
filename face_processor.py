"""
Core processing module for face reconstruction and editing.
Handles the main pipeline from input image to 3D reconstruction.
"""

import time
import torch
import numpy as np
import dlib
from PIL import Image
import torchvision.transforms as transforms
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import logging

# For Debugging
from utils.common import tensor2im

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ProcessingOptions:
    pose_range: Tuple[int, int]
    center_pose: int
    show_generated: bool
    show_multipose: bool
    show_3d: bool

class FaceProcessor:
    def __init__(self, config):
        """
        Initialize the face processor with configuration.
        
        Args:
            config: Configuration object containing all necessary paths and parameters
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self._initialize_models()
        self._setup_transforms()
        
    def _setup_transforms(self):
        """Setup image transformations for the model."""
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def _initialize_models(self):
        """Initialize all required models and components."""
        try:
            from models.stylegan3.model import GeneratorType
            from FaceBoxes.FaceBoxes_ONNX import FaceBoxes_ONNX
            from TDDFA.TDDFA_ONNX import TDDFA_ONNX
            from editing.interfacegan.face_editor import FaceEditor
            from utils.inference_utils import load_encoder, get_average_image
            
            logger.info("Initializing face detection and alignment models...")
            self.face_detector = dlib.get_frontal_face_detector()
            self.shape_predictor = dlib.shape_predictor(str(self.config.shape_predictor_path))
            
            logger.info("Initializing 3DDFA models...")
            self.face_boxes = FaceBoxes_ONNX()
            self.tddfa = TDDFA_ONNX(**self.config.threeddfa_config)
            
            logger.info("Loading encoder model...")
            self.encoder, self.opts = load_encoder(
                checkpoint_path=str(self.config.e4e_model_path)
            )
            self.encoder = self.encoder.to(self.device)
            self.encoder.eval()
            
            logger.info("Initializing face editor...")
            self.face_editor = FaceEditor(
                stylegan_generator=self.encoder.decoder,
                generator_type=GeneratorType.ALIGNED
            )
            
            self.avg_image = get_average_image(self.encoder)
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize models: {str(e)}")
    
    def process_image(self, input_path: str, options: ProcessingOptions) -> Dict[str, Path]:
        """
        Process input image through the complete pipeline.
        
        Args:
            input_path: Path to input image
            options: Processing options including pose range and visualization flags
            
        Returns:
            Dictionary containing paths to generated files
        """
        try:
            logger.info(f"Processing image: {input_path}")
            
            # Load and preprocess image
            original_image = Image.open(input_path).convert("RGB")
            aligned_image = self._align_face(original_image)
            cropped_image = self._crop_face(original_image)
            
            # Get transform matrix
            transform_matrix = self._compute_transform(aligned_image, cropped_image)
            
            # Generate inversions and edits
            result_batch, result_latents = self._generate_inversions(aligned_image)
            edit_images = self._generate_pose_edits(
                result_latents,
                transform_matrix,
                options.pose_range
            )
            
            # Generate 3D reconstruction
            frontal_face = edit_images[options.center_pose]
            output_paths = self._generate_3d_reconstruction(frontal_face)
            
            # Save pose images
            output_paths['pose'] = self._save_pose_images(edit_images)
            
            if options.show_3d:
                self._visualize_3d(output_paths['obj'])
                
            return output_paths
            
        except Exception as e:
            raise RuntimeError(f"Failed to process image: {str(e)}")
    
    def _align_face(self, image: Image.Image) -> Image.Image:
        """
        Align face in image using facial landmarks.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Aligned PIL Image
        """
        from utils.alignment_utils import align_face
        
        logger.info("Aligning face...")
        try:
            aligned_image = align_face(
                image,
                self.face_detector,
                self.shape_predictor
            )
            return aligned_image
        except Exception as e:
            raise RuntimeError(f"Face alignment failed: {str(e)}")
    
    def _crop_face(self, image: Image.Image) -> Image.Image:
        """
        Crop face from image using facial landmarks.
        
        Args:
            image: Input PIL Image
        
        Returns:
            Cropped PIL Image
        """
        from utils.alignment_utils import crop_face
        
        logger.info("Cropping face...")
        try:
            cropped_image = crop_face(
                image,
                self.face_detector,
                self.shape_predictor
            )
            return cropped_image
        except Exception as e:
            raise RuntimeError(f"Face cropping failed: {str(e)}")
    
    def _compute_transform(self, aligned_image: Image.Image, cropped_image: Image.Image) -> np.ndarray:
        """
        Compute transformation matrix between aligned and cropped images.
        
        Args:
            aligned_image: Aligned face image
            cropped_image: Cropped face image
            
        Returns:
            Transformation matrix
        """
        from utils.alignment_utils import get_stylegan_transform
        
        logger.info("Computing transformation matrix...")
        try:
            res = get_stylegan_transform(
                cropped_image,
                aligned_image,
                self.face_detector,
                self.shape_predictor
            )
            
            if res is None:
                raise RuntimeError("Failed to compute transformation matrix")
                
            rotation_angle, translation, transform, inverse_transform = res
            return inverse_transform
            
        except Exception as e:
            raise RuntimeError(f"Transform computation failed: {str(e)}")
    
    def _generate_inversions(self, image: Image.Image) -> Tuple[torch.Tensor, np.ndarray]:
        """
        Generate StyleGAN inversions for the input image.
        
        Args:
            image: Input PIL Image
            
        Returns:
            Tuple of (result_batch, result_latents)
        """
        from utils.inference_utils import run_on_batch
        
        logger.info("Generating inversions...")
        try:
            transformed_image = self.transform(image)
            transformed_image = transformed_image.unsqueeze(0).to(self.device).float()
            
            self.opts.n_iters_per_batch = 3
            self.opts.resize_outputs = False
            
            with torch.no_grad():
                tic = time.time()
                result_batch, result_latents = run_on_batch(
                    inputs=transformed_image,
                    net=self.encoder,
                    opts=self.opts,
                    avg_image=self.avg_image
                )
                toc = time.time()
                logger.info(f'Inference took {toc - tic:.4f} seconds')
            
            #Debugging
            # result_tensors = result_batch[0]
            # inversed_img = tensor2im(result_tensors[-1])
            # inversed_img.save("wakao.png")
                
            return result_batch, result_latents
            
        except Exception as e:
            raise RuntimeError(f"Inversion generation failed: {str(e)}")
    
    def _generate_pose_edits(
        self,
        latents: np.ndarray,
        transform_matrix: np.ndarray,
        pose_range: Tuple[int, int]
    ) -> List[Image.Image]:
        """
        Generate pose variations using the face editor.
        
        Args:
            latents: Input latent codes
            transform_matrix: Transformation matrix
            pose_range: Tuple of (min_pose, max_pose)
            
        Returns:
            List of edited images
        """
        logger.info("Generating pose variations...")
        try:
            input_latent = torch.from_numpy(latents[0][-1]).unsqueeze(0).to(self.device)
            # print(type(transform_matrix))
            edit_images, edit_latents = self.face_editor.edit(
                latents=input_latent,
                direction='pose',
                factor_range=pose_range,
                user_transforms=transform_matrix,
                apply_user_transformations=True
            )
            
            return [image[0] for image in edit_images]
            
        except Exception as e:
            raise RuntimeError(f"Pose editing failed: {str(e)}")
    
    def _generate_3d_reconstruction(self, image: Image.Image) -> Dict[str, Path]:
        """
        Generate 3D reconstruction using 3DDFA.
        
        Args:
            image: Input frontal face image
            
        Returns:
            Dictionary containing paths to generated files
        """
        from ThreeDDFA_utils.uv import uv_tex
        from ThreeDDFA_utils.serialization import ser_to_obj
        
        logger.info("Generating 3D reconstruction...")
        try:
            # Convert to BGR for 3DDFA
            image_np = np.array(image)
            image_bgr = image_np[:, :, ::-1]
            
            # Detect face and estimate parameters
            boxes = self.face_boxes(image_bgr)
            param_lst, roi_box_lst = self.tddfa(image_bgr, boxes)
            ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=True)
            
            # Generate output paths
            output_paths = self.config.get_output_paths()
            
            # Generate UV texture and OBJ file
            uv_tex(image_bgr, ver_lst, self.tddfa.tri, wfp=str(output_paths['uv_tex']))
            ser_to_obj(
                image_bgr,
                ver_lst,
                self.tddfa.tri,
                height=1024,
                wfp=str(output_paths['obj'])
            )
            
            return output_paths
            
        except Exception as e:
            raise RuntimeError(f"3D reconstruction failed: {str(e)}")
    
    def _save_pose_images(self, images: List[Image.Image]) -> Path:
        """
        Save multi-pose visualization.
        
        Args:
            images: List of pose variations
            
        Returns:
            Path to saved image
        """
        logger.info("Saving pose variations...")
        try:
            # Create concatenated image
            res = np.array(images[0].resize((512, 512)))
            for image in images[1:]:
                res = np.concatenate([res, image.resize((512, 512))], axis=1)
            
            # Save result
            output_paths = self.config.get_output_paths()
            pose_img = Image.fromarray(res).convert("RGB")
            pose_img.save(str(output_paths['pose']))
            
            return output_paths['pose']
            
        except Exception as e:
            raise RuntimeError(f"Failed to save pose images: {str(e)}")
    
    def _visualize_3d(self, obj_path: Path):
        """
        Visualize 3D reconstruction using Open3D.
        
        Args:
            obj_path: Path to OBJ file
        """
        try:
            import open3d as o3d
            logger.info("Visualizing 3D reconstruction...")
            
            mesh = o3d.io.read_triangle_mesh(str(obj_path))
            o3d.visualization.draw_geometries([mesh])
            
        except ImportError:
            logger.warning("Open3D not available. Skipping 3D visualization.")
        except Exception as e:
            logger.error(f"3D visualization failed: {str(e)}")