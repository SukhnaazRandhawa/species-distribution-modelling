"""Sanity checks for the data loading code."""
import sys
sys.path.append('.')            # lets Python find the src package
from src.data import load_all

d = load_all()
Y = d['Y_test']

print('Train locations:', d['train_locs'].shape)
print('Test locations: ', d['test_locs'].shape)
print('Answer key:     ', Y.shape, Y.dtype)
print(f'Mean presence rate: {Y.mean()*100:.2f}%')
print('Species with no test presences:', (Y.sum(0) == 0).sum())
print('Test locations with no species:', (Y.sum(1) == 0).sum())
print('Species per test location (mean / max):',
      round(Y.sum(1).mean(), 1), Y.sum(1).max())