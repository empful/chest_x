跑训练的代码：
python retrain.py

主要代码是 model.py (模型＋训练的代码), cxr_dataset.py (dataset代码)， eval_model.py（算AUC代码）。

原本densenet 是用model.py 里的 train_cnn 函数训练的， dataset是 CXRDataset
我们patches 方法是用model.py里 train_cnn_split2 函数训练的, dataset是 CXRDataset3
