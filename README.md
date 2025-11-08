### PicoLLM

Install requirements:
```bash
pip install -r requirements.txt
```
Sanity check command:

```bash
python pico-llm.py --block_size 32 --tinystories_weight 0.0 --input_files 3seqs.txt --prompt "0 1 2 3 4"
```

Running Transformer sanity check
```bash
python pico-llm.py \
  --device_id cpu \
  --embed_size 512 \
  --block_size 256 \
  --transformer_n_blocks 6 \
  --transformer_n_heads 8 \
  --use_kv_cache \
  --prompt "Once upon a time"
  ```
