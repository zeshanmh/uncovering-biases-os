python3 main.py --bias_S --bias_Y1 --bias_type "selection_bias" --d 2 3 &
python3 main.py --bias_A --bias_Y1 --bias_type "confounding_bias" --d 2 3 &
python3 main.py --bias_trs --bias_Y1 --bias_type "transportability_bias" --d 2 3 &
python3 main.py --bias_S --bias_A --bias_trs --bias_Y1 --bias_type "all_bias_full_adj" --d 2 3 
