python AmpGenix.py \
  --device 0 \
  --ntokens 8-15 \
  --nsamples 100 \
  --model_path /AMP_models/AmpGenix/ \
  --prefix "S" \
  --topp 5 \
  --temperature 1 \
  --save_samples \
  --save_samples_path /Data/samples_saved/prefix_S/
