import os
print("Installing spotlight", flush=True)
os.system("conda install -y -c maciejkula spotlight")

print("Success!", flush=True)
from spotlight.cross_validation import random_train_test_split
from spotlight.datasets.movielens import get_movielens_dataset
from spotlight.evaluation import rmse_score
from spotlight.factorization.explicit import ExplicitFactorizationModel
print('making model for spotlight', flush=True)

dataset = get_movielens_dataset(variant='100K')

train, test = random_train_test_split(dataset)

model = ExplicitFactorizationModel(n_iter=1)
model.fit(train)

rmse = rmse_score(model, test)
print(rmse, flush=True)
