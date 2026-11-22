import numpy as np

train = np.load('species_train.npz')
extra = np.load('species_train_extra.npz')
test = np.load('species_test.npz', allow_pickle=True)

_, counts = np.unique(train['train_ids'], return_counts=True)
print('Median observations per species:', np.median(counts))
print('Species with exactly 2000 observations:', (counts == 2000).sum())

print('Extra training observations:', extra['train_locs'].shape[0])

n_test = test['test_locs'].shape[0]
pos = [len(p) / n_test * 100 for p in test['test_pos_inds']]
print(f'Avg % of test locations where a species is present: {np.mean(pos):.2f}%')