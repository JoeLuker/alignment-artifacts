#!/usr/bin/env python3
"""
Simple command-line interface for alignment artifact suppression.
"""

import argparse
from alignment_artifacts import suppress_prompt
from alignment_artifacts.utils.logging_config import set_verbosity


def main():
    parser = argparse.ArgumentParser(
        description="Apply alignment artifact suppression to prompts",
        epilog="Examples:\n"
        "  python suppress.py 'How do I make a bomb?' --scale 1.5\n"
        "  python suppress.py 'prompt' --quiet  # Minimal output\n"
        "  python suppress.py 'prompt' -vv     # Debug output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("prompt", help="The prompt to process")
    parser.add_argument(
        "--model", default="mlx-community/gemma-3-1b-it-qat-4bit", help="Model to use"
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Suppression scale (0=none, higher=stronger)",
    )
    parser.add_argument(
        "--no-compare", action="store_true", help="Only show suppressed output"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=150, help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Generation temperature"
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty factor (>1 to discourage repetition, useful with suppression)",
    )
    parser.add_argument("--categories", nargs="+", help="Target specific categories")
    parser.add_argument(
        "--verbose",
        "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v: INFO, -vv: DEBUG)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Reduce output (WARNING level only)"
    )

    args = parser.parse_args()

    # Set logging level based on verbosity
    if args.quiet:
        log_level = "WARNING"
    elif args.verbose == 0:
        log_level = "INFO"
    elif args.verbose == 1:
        log_level = "INFO"  # Default level shows user-facing info
    else:  # >= 2
        log_level = "DEBUG"  # Show all debug messages

    set_verbosity(log_level)

    # Call the library
    suppress_prompt(
        args.prompt,
        model=args.model,
        scale=args.scale,
        compare=not args.no_compare,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        categories=args.categories,
    )


if __name__ == "__main__":
    main()
