python Fine-tuning_classifier.py \
      --raw_data_path /Data/train_dat.csv \
      --batch_size 512 \
      --epochs 20 \
      --model_path /AMP_models/ProteoGPT/ \
      --output_path /Classifier/best_model.pt \
      --trainset_path /Classifier/trainset.csv \
      --validset_path /Classifier/validset.csv \
      --testset_path /Classifier/testset.csv \
      --loss_path /Classifier/loss.png \
      --acc_path /Classifier/acc.png \
      --metrix_path /Classifier/matrix.png \
      --roc_path /Classifier/roc.png \
      --report_path /Classifier/report.txt \
      --predict_path /Classifier/predicted_result.csv \
      --allloss_path /Classifier/allloss.csv \
      --allacc_path /Classifier/allacc.csv \
      --auc_path /Classifier/auc.csv \






