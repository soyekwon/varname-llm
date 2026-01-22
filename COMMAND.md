### summarize_v

python dapt_llama3_from_docx.py summarize \
  --input_dir /path/to/word_files \
  --output_jsonl /path/to/out/train.jsonl \
  --large_mode \
  --chunk_chars 10000 \
  --preserve_specs

### train

python dapt_llama3_from_docx.py train   —model_id ./model_path   —train_jsonl ./vehicle_specs_summ.jsonl   —output_dir ./llama3_vehicle_dapt_lora   —use_lora —use_qlora  —bf16 —gradient_checkpointing   —eval_ratio 0.1
