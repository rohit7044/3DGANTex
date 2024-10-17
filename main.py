"""
Main script for running the 3D face reconstruction pipeline.
"""

import argparse
from pathlib import Path
from config import Config
from face_processor import FaceProcessor, ProcessingOptions

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="3D Face Reconstruction and Editing")
    
    parser.add_argument("--input", type=str, required=True,
                      help="Path to input image file (Consider using full path if file not found error persists)")
    parser.add_argument("--pose-min", type=float, default=-3,
                      help="Minimum pose angle")
    parser.add_argument("--pose-max", type=float, default=5,
                      help="Maximum pose angle")
    parser.add_argument("--center-pose", type=int, default=1,
                      help="Index of center pose")
    parser.add_argument("--output-dir", type=str, default="output",
                      help="Output directory (Consider using full path and it will create a directory)")
    parser.add_argument("--show-generated", action="store_true",
                      help="Show generated images ")
    parser.add_argument("--show-multipose", action="store_true",
                      help="Show multipose images")
    parser.add_argument("--show-3d", action="store_true",
                      help="Show 3D visualization")
    
    return parser.parse_args()

def main():
    """Main execution function."""
    try:
        # Parse arguments
        args = parse_arguments()

        # Get input directory for string processing and filename creation
        input_dir = args.input

        # Create output directory if not available
        output_dir = args.output_dir
        
        # Initialize configuration
        config = Config(input_dir,output_dir)
        
        # Initialize processor
        processor = FaceProcessor(config)
        
        # Create processing options
        options = ProcessingOptions(
            pose_range=(int(args.pose_min), int(args.pose_max)),
            center_pose=args.center_pose,
            show_generated=args.show_generated,
            show_multipose=args.show_multipose,
            show_3d=args.show_3d
        )
        
        # Process image
        results = processor.process_image(args.input, options)
        
        print("Processing completed successfully!")
        print("Generated files:")
        for key, path in results.items():
            print(f"  {key}: {path}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0

if __name__ == "__main__":
    exit(main())