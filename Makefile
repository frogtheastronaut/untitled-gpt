DATA_FILES = data/TinyStories-train.txt
WORK_DIR = out

# 6M Parameters
# n_embd=288, n_layer=6, n_head=6
train_6m:
	python train.py \
		--data_files $(DATA_FILES) \
		--work_dir $(WORK_DIR)/6m \
		--separator "<|endoftext|>" \
		--max_lines 1000 \
		--n_layer 6 \
		--n_head 6 \
		--n_embd 288 \
		--block_size 128 \
		--batch_size 32 \
		--max_iters 5000 \
		--resume

# 20M Parameters
# n_embd=420, n_layer=10, n_head=10
train_20m:
	python train.py \
		--data_files $(DATA_FILES) \
		--work_dir $(WORK_DIR)/20m \
		--n_layer 10 \
		--n_head 10 \
		--n_embd 420 \
		--block_size 256 \
		--batch_size 16 \
		--max_iters 10000 \
		--resume

# 50M Parameters
# n_embd=512, n_layer=16, n_head=8
train_50m:
	python train.py \
		--data_files $(DATA_FILES) \
		--work_dir $(WORK_DIR)/50m \
		--n_layer 16 \
		--n_head 8 \
		--n_embd 512 \
		--block_size 256 \
		--batch_size 8 \
		--max_iters 20000 \
		--resume

# 100M Parameters
# n_embd=768, n_layer=14, n_head=12
train_100m:
	python train.py \
		--data_files $(DATA_FILES) \
		--work_dir $(WORK_DIR)/100m \
		--n_layer 14 \
		--n_head 12 \
		--n_embd 768 \
		--block_size 512 \
		--batch_size 4 \
		--max_iters 50000 \
		--resume

clean:
	rm -rf $(WORK_DIR)
