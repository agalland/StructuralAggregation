from train import StructAgg
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset_name = "DD"

args = parse_args()

num_layers = int(args["num_layers"])
num_epochs = int(args["num_epochs"])
verbose = args["verbose"]
reg = args["reg"]
path_results = str(args["path_results"])
path_weights = str(args["path_weights"])
path_data = str(args["path_data"])
lr = float(args["lr"])
bins = int(args["bins"])
n_dim = int(args["n_dim"])
alpha_reg = float(args["alpha_reg"])

net = StructAgg

# Load dataset
print('Load dataset {}'.format(dataset_name))
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
k = 0
g_size = X[0][1].shape[0]
out_dim = int(np.max(y)) + 1

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
