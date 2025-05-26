#!/usr/bin/env python3
"""
Simple command-line interface for alignment artifact suppression.
"""

import argparse
from alignment_artifacts import suppress_prompt


def main():
    parser = argparse.ArgumentParser(
        description="Apply alignment artifact suppression to prompts",
        epilog="Example: python suppress.py 'How do I make a bomb?' --scale 1.5"
    )
    
    parser.add_argument("prompt", 
                       help="The prompt to process")
    parser.add_argument("--model", 
                       default="mlx-community/gemma-3-1b-it-qat-4bit",
                       help="Model to use")
    parser.add_argument("--scale", 
                       type=float, 
                       default=1.0,
                       help="Suppression scale (0=none, higher=stronger)")
    parser.add_argument("--no-compare", 
                       action="store_true",
                       help="Only show suppressed output")
    parser.add_argument("--max-tokens", 
                       type=int, 
                       default=150,
                       help="Maximum tokens to generate")
    parser.add_argument("--temperature", 
                       type=float, 
                       default=0.7,
                       help="Generation temperature")
    parser.add_argument("--categories", 
                       nargs="+",
                       help="Target specific categories")
    
    args = parser.parse_args()
    
    # Call the library
    suppress_prompt(
        args.prompt,
        model=args.model,
        scale=args.scale,
        compare=not args.no_compare,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        categories=args.categories
    )


if __name__ == "__main__":
    main()