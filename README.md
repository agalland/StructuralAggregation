# Graph Structural Aggregation For Explainable Learning

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```


## Training

To train the model(s) in the paper, run this command:

```train
python train_model.py --datasetName="DD" --lr=0.005 --numLayers=1 --alphaReg=0.0001 --ndim=64 --bins=5 --reg=True --pathData="data/" --pathWeights="weights/" --pathResults="results/" --verbose=True
```

The setting for the split train, test, validation is the same as the one described in the experiments. The hyper-parameters selected are the one that produce the best classification accuracy.


## Evaluation

To evaluate my model on DD, run:

```eval
python eval.py --datasetName="DD" --lr=0.005 --numLayers=1 --alphaReg=0.0001 --ndim=64 --bins=5 --pathData="data/" --loadWeights="weightsEval/" --verbose=True
```

A random fold was selected and train on the molecules of the train index. The accuracies are displayed for the train, test and validation sets.
Train, test and validation indices are stores in "weightsEval/" as well.

## Pre-trained Models

You can download pretrained models here:

- weights are included in the zip file in the weightsEval folder


## Results

Our model achieves the following performance on graph classification datasets:

### [Graph Classification on PROTEINS, DD and COLLAB]

| Model name  |     PROTEINS    |      DD     |    COLLAB   |
| ------------|---------------- | ----------- | ----------- |
| structAgg   |     76.72%      |   78.42%    |   80.26%    |



