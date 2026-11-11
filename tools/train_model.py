import argparse
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from training.engine import train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Train a model against a frozen research data release')
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--max-steps',type=int)
    args=parser.parse_args()
    if args.max_steps is not None and args.max_steps<1:parser.error('--max-steps must be positive')
    train(args.config,ROOT,args.resume,args.max_steps)
