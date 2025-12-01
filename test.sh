### test
python3 main.py --save_dir ./test/demo/output/ \
               --reset True \
               --log_file_name test.log \
               --test True \
               --num_workers 1 \
               --lr_vis_path ./test/demo/lr_vis/dji-matrice600.png \
               --lr_thr_path ./test/demo/lr_thr/dji-matrice600.png \
               --ref_vis_path ./test/demo/ref_vis/dji-matrice600.png \
               --ref_thr_path ./test/demo/ref_thr/dji-matrice600.png\
               --model_path ./model.pt
