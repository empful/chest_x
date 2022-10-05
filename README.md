# chest_x

Generate jsonl file with labels
```
python process_labels.py
```


Run fine-tune and evaluate
```
python -m torch.distributed.launch --nproc_per_node=2 vit_main.py
```
