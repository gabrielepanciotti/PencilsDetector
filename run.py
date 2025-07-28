"""
Entry point script for running pencil detection.
"""
import os
import cv2
import argparse

from image_utils.loader import load_image
from main import process_image, detect_green_pencils, detect_all_pencils, visualize_results
from results.exporter import generate_results_json
from results.summary import print_detection_summary, print_debug_info

def main():
    """Main function to run the pencil detection pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Detect and count pencils by color')
    parser.add_argument('--image', type=str, required=True, help='Path to the input image')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory to save output files')
    parser.add_argument('--show', action='store_true', help='Show visualization')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with additional visualizations')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create debug directory if debug mode is enabled
    if args.debug:
        debug_dir = os.path.join(os.getcwd(), "debug")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"Debug mode enabled. Debug images will be saved to {debug_dir}")
    
    # Load image
    image = load_image(args.image)
    print(f"Loaded image with shape: {image.shape}")
    
    # Save original image for reference if in debug mode
    if args.debug:
        cv2.imwrite(os.path.join(debug_dir, "original.jpg"), image)
    
    # Process image once for all detection steps
    hsv, blurred, edge_mask, segmentation_mask = process_image(image, debug=args.debug)
    
    # Detect green pencils (Task 1 & 2)
    green_pencils = detect_green_pencils(image, hsv=hsv, debug=args.debug)
    
    # Detect all pencils (Task 3)
    all_pencils = detect_all_pencils(image, hsv=hsv, edge_mask=edge_mask, segmentation_mask=segmentation_mask, debug=args.debug)
    
    # Generate visualization
    output_image_path = os.path.join(args.output_dir, 'pencils_detected.jpg')
    vis_image = visualize_results(image, green_pencils, all_pencils, output_image_path)
    
    # Generate JSON results
    output_json_path = os.path.join(args.output_dir, 'pencils_results.json')
    results = generate_results_json(green_pencils, all_pencils, output_json_path)
    
    # Print summary
    print_detection_summary(green_pencils, all_pencils)
    
    # Print HSV values for debugging color classification if in debug mode
    if args.debug:
        print_debug_info(all_pencils)
    
    # Show visualization if requested
    if args.show:
        cv2.imshow('Pencil Detection', vis_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
