### evaluation
python3 main.py --save_dir ./eval/DRONE/ThermoVisSR \
               --reset True \
               --log_file_name eval.log \
               --eval True \
               --eval_save_results True \
               --num_workers 4 \
               --dataset DRONES \
               --dataset_dir DRONES/ \
               --model_path ./model.pt
