"""
Bahlib CLI - Command Line Interface for Face Anonymization

Usage:
    bahlib image input.jpg -o output.jpg --blur 51
    bahlib video input.mp4 -o output.mp4
    bahlib webcam --blur 99
    bahlib batch ./photos ./anonymized
"""

import argparse
import sys
import cv2
from pathlib import Path

from .core import Bahlib
from .video import anonymize_video, anonymize_webcam
from .batch import anonymize_directory


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="bahlib",
        description="Bahlib - 100% Local Face Anonymization Tool",
        epilog="Your data never leaves your machine. Privacy is king!"
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version="%(prog)s 0.1.0"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # ==================== IMAGE SUBCOMMAND ====================
    image_parser = subparsers.add_parser(
        "image",
        help="Anonymize a single image"
    )
    image_parser.add_argument(
        "input",
        help="Path to input image"
    )
    image_parser.add_argument(
        "-o", "--output",
        help="Path to output image (default: input_anonymized.ext)"
    )
    image_parser.add_argument(
        "--blur", "-b",
        type=int,
        default=51,
        help="Blur strength (odd number, default: 51)"
    )
    image_parser.add_argument(
        "--feather", "-f",
        type=float,
        default=0.3,
        help="Edge feather amount 0.0-1.0 for smooth blending (default: 0.3)"
    )
    image_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0.0-1.0 (default: 0.5)"
    )
    image_parser.add_argument(
        "--model", "-m",
        type=int,
        choices=[0, 1],
        default=1,
        help="Model selection: 0=short-range, 1=full-range (default: 1)"
    )
    image_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Upscale factor for small face detection (e.g., 1.5, 2.0)"
    )
    image_parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale detection for group photos with small faces"
    )
    image_parser.add_argument(
        "--tiled",
        action="store_true",
        help="Use tiled detection for group photos (best for many small faces)"
    )
    image_parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Tile size for tiled detection (default: 640)"
    )
    image_parser.add_argument(
        "--method",
        choices=['blur', 'pixelate', 'blackbar', 'all'],
        default='blur',
        help="Anonymization method: blur, pixelate, blackbar, or all (default: blur)"
    )
    image_parser.add_argument(
        "--pixelate-blocks",
        type=int,
        default=10,
        help="Number of pixel blocks for pixelation (default: 10, lower = more pixelated)"
    )
    image_parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Display the result in a window"
    )
    
    # ==================== VIDEO SUBCOMMAND ====================
    video_parser = subparsers.add_parser(
        "video",
        help="Anonymize a video file"
    )
    video_parser.add_argument(
        "input",
        help="Path to input video"
    )
    video_parser.add_argument(
        "-o", "--output",
        required=True,
        help="Path to output video"
    )
    video_parser.add_argument(
        "--blur", "-b",
        type=int,
        default=51,
        help="Blur strength (odd number, default: 51)"
    )
    video_parser.add_argument(
        "--feather", "-f",
        type=float,
        default=0.3,
        help="Edge feather amount 0.0-1.0 for smooth blending (default: 0.3)"
    )
    video_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0.0-1.0 (default: 0.5)"
    )
    video_parser.add_argument(
        "--model", "-m",
        type=int,
        choices=[0, 1],
        default=1,
        help="Model selection: 0=short-range, 1=full-range (default: 1)"
    )
    video_parser.add_argument(
        "--codec",
        default="mp4v",
        help="Video codec (default: mp4v)"
    )
    
    # ==================== WEBCAM SUBCOMMAND ====================
    webcam_parser = subparsers.add_parser(
        "webcam",
        help="Run live face anonymization on webcam"
    )
    webcam_parser.add_argument(
        "--camera", "-i",
        type=int,
        default=0,
        help="Camera device ID (default: 0)"
    )
    webcam_parser.add_argument(
        "--blur", "-b",
        type=int,
        default=51,
        help="Blur strength (odd number, default: 51)"
    )
    webcam_parser.add_argument(
        "--feather", "-f",
        type=float,
        default=0.3,
        help="Edge feather amount 0.0-1.0 for smooth blending (default: 0.3)"
    )
    webcam_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0.0-1.0 (default: 0.5)"
    )
    webcam_parser.add_argument(
        "--model", "-m",
        type=int,
        choices=[0, 1],
        default=1,
        help="Model selection: 0=short-range, 1=full-range (default: 1)"
    )
    webcam_parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Disable horizontal flip (mirror effect)"
    )
    
    # ==================== BATCH SUBCOMMAND ====================
    batch_parser = subparsers.add_parser(
        "batch",
        help="Anonymize all images in a directory"
    )
    batch_parser.add_argument(
        "input_dir",
        help="Path to input directory"
    )
    batch_parser.add_argument(
        "output_dir",
        help="Path to output directory"
    )
    batch_parser.add_argument(
        "--blur", "-b",
        type=int,
        default=51,
        help="Blur strength (odd number, default: 51)"
    )
    batch_parser.add_argument(
        "--feather", "-f",
        type=float,
        default=0.3,
        help="Edge feather amount 0.0-1.0 for smooth blending (default: 0.3)"
    )
    batch_parser.add_argument(
        "--confidence", "-c",
        type=float,
        default=0.5,
        help="Detection confidence threshold 0.0-1.0 (default: 0.5)"
    )
    batch_parser.add_argument(
        "--model", "-m",
        type=int,
        choices=[0, 1],
        default=1,
        help="Model selection: 0=short-range, 1=full-range (default: 1)"
    )
    batch_parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Process subdirectories recursively"
    )
    batch_parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Upscale factor for small face detection (e.g., 1.5, 2.0)"
    )
    batch_parser.add_argument(
        "--multi-scale",
        action="store_true",
        help="Enable multi-scale detection for group photos with small faces"
    )
    batch_parser.add_argument(
        "--tiled",
        action="store_true",
        help="Use tiled detection for group photos (best for many small faces)"
    )
    batch_parser.add_argument(
        "--tile-size",
        type=int,
        default=640,
        help="Tile size for tiled detection (default: 640)"
    )
    batch_parser.add_argument(
        "--method",
        choices=['blur', 'pixelate', 'blackbar', 'all'],
        default='blur',
        help="Anonymization method: blur, pixelate, blackbar, or all (default: blur)"
    )
    batch_parser.add_argument(
        "--pixelate-blocks",
        type=int,
        default=10,
        help="Number of pixel blocks for pixelation (default: 10, lower = more pixelated)"
    )
    
    return parser


def cmd_image(args) -> int:
    """Handle the image subcommand."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        stem = input_path.stem
        suffix = input_path.suffix
        output_path = str(input_path.parent / f"{stem}_anonymized{suffix}")
    
    print(f"Processing: {args.input}")
    
    try:
        with Bahlib(
            model_selection=args.model,
            min_detection_confidence=args.confidence
        ) as bh:
            result = bh.anonymize(
                args.input, 
                blur_strength=args.blur,
                feather_amount=args.feather,
                scale_factor=args.scale,
                multi_scale=args.multi_scale,
                tiled=args.tiled,
                tile_size=args.tile_size,
                method=args.method,
                pixelate_blocks=args.pixelate_blocks
            )
            cv2.imwrite(output_path, result)
            print(f"Saved: {output_path}")
            
            if args.show:
                cv2.imshow("Bahlib Result - Press any key to close", result)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_video(args) -> int:
    """Handle the video subcommand."""
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        return 1
    
    print(f"Processing video: {args.input}")
    print("This may take a while...")
    
    def progress(current, total):
        pct = (current / total) * 100
        print(f"\rProgress: {current}/{total} frames ({pct:.1f}%)", end="", flush=True)
    
    try:
        stats = anonymize_video(
            input_path=args.input,
            output_path=args.output,
            blur_strength=args.blur,
            feather_amount=args.feather,
            model_selection=args.model,
            min_detection_confidence=args.confidence,
            codec=args.codec,
            progress_callback=progress
        )
        
        print()  # Newline after progress
        print(f"Done! Processed {stats['processed_frames']} frames")
        print(f"Resolution: {stats['resolution'][0]}x{stats['resolution'][1]} @ {stats['fps']} FPS")
        print(f"Saved: {args.output}")
    
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    
    return 0


def cmd_webcam(args) -> int:
    """Handle the webcam subcommand."""
    print("Starting webcam anonymization...")
    print("Press ESC to quit")
    
    try:
        anonymize_webcam(
            camera_id=args.camera,
            blur_strength=args.blur,
            feather_amount=args.feather,
            model_selection=args.model,
            min_detection_confidence=args.confidence,
            mirror=not args.no_mirror
        )
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    return 0


def cmd_batch(args) -> int:
    """Handle the batch subcommand."""
    input_dir = Path(args.input_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}", file=sys.stderr)
        return 1
    
    print(f"Processing directory: {args.input_dir}")
    if args.recursive:
        print("(recursive mode enabled)")
    
    def progress(filename, current, total):
        pct = (current / total) * 100
        print(f"[{current}/{total}] ({pct:.1f}%) {filename}")
    
    try:
        stats = anonymize_directory(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            blur_strength=args.blur,
            feather_amount=args.feather,
            model_selection=args.model,
            min_detection_confidence=args.confidence,
            recursive=args.recursive,
            progress_callback=progress,
            scale_factor=args.scale,
            multi_scale=args.multi_scale,
            tiled=args.tiled,
            tile_size=args.tile_size,
            method=args.method,
            pixelate_blocks=args.pixelate_blocks
        )
        
        print()
        print("=" * 40)
        print(f"Total images found: {stats['total']}")
        print(f"Successfully processed: {len(stats['processed'])}")
        
        if stats['failed']:
            print(f"Failed: {len(stats['failed'])}")
            for filepath, error in stats['failed']:
                print(f"  - {filepath}: {error}")
        
        print(f"Output directory: {args.output_dir}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    
    return 0


def main() -> int:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command == "image":
        return cmd_image(args)
    elif args.command == "video":
        return cmd_video(args)
    elif args.command == "webcam":
        return cmd_webcam(args)
    elif args.command == "batch":
        return cmd_batch(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

