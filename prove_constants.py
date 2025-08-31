#!/usr/bin/env python3
"""
prove_constants.py - Stage 1 of the unified proof pipeline

This script runs the RG step analysis and exports constants for Stage 2.
Usage:
    python prove_constants.py --output ../ym_bounds/input_constants.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Stage 1: Prove RG constants from step analysis"
    )
    
    # RG analysis parameters
    parser.add_argument("--T", type=int, default=4, help="Temporal lattice size")
    parser.add_argument("--L", type=int, default=4, help="Spatial lattice size") 
    parser.add_argument("--n-cfg", type=int, default=24, help="Number of configurations")
    parser.add_argument("--b-space", type=int, default=2, help="Spatial blocking factor")
    parser.add_argument("--b-time", type=int, default=2, help="Temporal blocking factor")
    parser.add_argument("--tau", type=float, default=0.2, help="Smoothing parameter")
    parser.add_argument("--alpha", type=float, default=0.6, help="KP alpha parameter")
    parser.add_argument("--gamma", type=float, default=0.6, help="KP gamma parameter")
    parser.add_argument("--rmax", type=int, default=2, help="KP radius maximum") 
    parser.add_argument("--seed", type=int, default=11, help="Random seed")
    
    # Output
    parser.add_argument("--output", type=Path, required=True, 
                       help="Output JSON file for constants (e.g., ../ym_bounds/input_constants.json)")
    
    args = parser.parse_args()
    
    print("🧮 Stage 1: Deriving RG constants from step analysis")
    print(f"   Parameters: T={args.T}, L={args.L}, configs={args.n_cfg}")
    print(f"   Blocking: space={args.b_space}, time={args.b_time}, τ={args.tau}")
    print(f"   Output: {args.output}")
    print()
    
    # Run the RG validator tool with export command
    cmd = [
        sys.executable, "rg_validator_tool.py", "export",
        "--T", str(args.T),
        "--L", str(args.L), 
        "--n-cfg", str(args.n_cfg),
        "--b-space", str(args.b_space),
        "--b-time", str(args.b_time),
        "--tau", str(args.tau),
        "--alpha", str(args.alpha),
        "--gamma", str(args.gamma),
        "--rmax", str(args.rmax),
        "--seed", str(args.seed),
        "--output", str(args.output)
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Warnings:", result.stderr)
        
        print(f"✅ Stage 1 complete! Constants exported to: {args.output}")
        print("\n🔗 Next step (Stage 2):")
        print(f"   cd ../ym_bounds")
        print(f"   python main.py --beta 6 --constants {args.output.name}")
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running RG analysis: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
