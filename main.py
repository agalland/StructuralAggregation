from train import StructAgg
from utils import *

import ssl


ssl._create_default_https_context = ssl._create_unverified_context
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# dataset_names = ["PROTEINS", "DD", "COLLAB"]
dataset_names = ["ogbg-molhiv"]

args = parse_args()

num_layers = int(args["num_layers"])
num_epochs = int(args["num_epochs"])
verbose = args["verbose"]
reg = args["reg"]
path_results = str(args["path_results"])
path_weights = str(args["path_weights"])
path_data = str(args["path_data"])

net = StructAgg

reg = True
params = []
for lr in [1e-2, 1e-3]:
    for num_layers in [1, 2]:
        for bins in [2, 5, 10]:
            for n_dim in [32, 64]:
                for alpha_reg in [0.0, 0.01]:
                    params.append((lr, num_layers, bins, n_dim, alpha_reg))

for dataset_name in dataset_names:
    train_idx = None
    test_idx = None
    val_idx = None
    if dataset_name.split("-")[0] != "ogbg":
        dataset_path = path_data + dataset_name + "/" + dataset_name
    else:
        dataset_path = dataset_name

    # Load dataset
    dataset = GraphDataset(dataset_path)
    try:
        train_idx = dataset.train_idx
        test_idx = dataset.valid_idx
        val_idx = dataset.test_idx
    except:
        pass

    X, y = normalize_adj(dataset)
    n_feat = X[0][1].shape[1]
    print("feature size: {}".format(n_feat))
    g_size = X[0][1].shape[0]
    out_dim = int(np.max(y)) + 1

    for lr, num_layers, bins, n_dim, alpha_reg in params:
        if not reg:
            alpha_reg = 0.0
        # Run 10 iterations of each set of parameter
        for k in range(0, 1):
            cross_val(net,
                      X,
                      y,
                      dataset_name,
                      path_results,
                      n_feat=n_feat,
                      n_dim=n_dim,
                      g_size=g_size,
                      bins=bins,
                      num_layers=num_layers,
                      out_dim=out_dim,
                      lr=lr,
                      num_epochs=num_epochs,
                      path_weights=path_weights,
                      device=device,
                      k=k,
                      verbose=verbose,
                      reg=reg,
                      alpha_reg=alpha_reg,
                      train_idx=train_idx,
                      test_idx=test_idx,
                      val_idx=val_idx)
            print("----------------")
            print("new set")
            print("----------------")
