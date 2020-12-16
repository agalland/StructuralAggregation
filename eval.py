from train import StructAgg
from utils import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


args = parse_args()

num_layers = int(args["num_layers"])
num_epochs = int(args["num_epochs"])
verbose = bool(args["verbose"])
reg = bool(args["reg"])
path_results = str(args["path_results"])
path_weights = str(args["path_weights"])
path_data = str(args["path_data"])
dataset_name = str(args["dataset_name"])

net = StructAgg

lr = float(args["lr"])
bins = int(args["bins"])
n_dim = int(args["n_dim"])
load_weights = str(args["path_load_weights"])

dataset_path = path_data + dataset_name + "/" + dataset_name

# Load dataset
dataset = GraphDataset(dataset_path)
X, y = normalize_adj(dataset)
n_feat = X[0][1].shape[1]
print("feature size: {}".format(n_feat))
g_size = X[0][1].shape[0]
out_dim = int(np.max(y))+1

index = 0

train_ind = np.load(load_weights + "train_ind.npy")
test_ind = np.load(load_weights + "test_ind.npy")
val_ind = np.load(load_weights + "val_ind.npy")

X_val = [X[k] for k in val_ind]
y_val = y[val_ind]
X_train = [X[k] for k in train_ind]
y_train = y[train_ind]
X_test = [X[k] for k in test_ind]
y_test = y[test_ind]

train_net = net(n_feat=n_feat,
                n_dim=n_dim,
                g_size=g_size,
                bins=bins,
                num_layers=num_layers,
                out_dim=out_dim,
                lr=lr,
                num_epochs=num_epochs,
                path_save_weights=None,
                device=device)
weights2load = load_weights + "weights"
train_net.net.load_state_dict(torch.load(weights2load))
acc_train, _ = train_net.predict(X_train, y_train)
acc_test, _ = train_net.predict(X_test, y_test)
acc_val, _ = train_net.predict(X_val, y_val)

print("Accuracies: train {}, test {}, val: {}".format(acc_train, acc_test, acc_val))
