import argparse

parser = argparse.ArgumentParser(description='Computing permutation feature importance in parallel.')

parser.add_argument("--climate", choices=["current", "future"], required=True, type=str, help="This is the current or future climate string choice.")
parser.add_argument("--boot_iterations", required=True, type=int, help="This is number of bootstrap iterations. Recommend 20.")
parser.add_argument("--seed_indexer", type=int, help="This is the index to pick iterations back up from where left off.")
parser.add_argument("--random_file", required=True, type=int, help="This is the random file to initialize.")

args=parser.parse_args()

print("made it thru parser")
exec(open('/glade/u/home/molina/python_scripts/deep-conus/deep-conus/08_dlmodel_evaluator.py').read())

which_climate=args.climate
which_random=args.random_file

file=EvaluateDLModel(climate=which_climate, method='random', variables=np.array(['TK','EV','EU','QVAPOR','PRESS']), 
                     var_directory=f'/glade/scratch/molina/DL_proj/{which_climate}_conus_fields/dl_preprocess/', 
                     model_directory=f'/glade/scratch/molina/DL_proj/current_conus_fields/dl_models/', 
                     model_num=25,
                     eval_directory=f'/glade/scratch/molina/DL_proj/{which_climate}_conus_fields/dl_models/', 
                     mask=False, mask_train=False, unbalanced=True, isotonic=False, bin_res=0.05,
                     random_choice=which_random, obs_threshold=0.5, print_sequential=False, 
                     perm_feat_importance=False, pfi_variable=None, pfi_iterations=None, currenttrain_futuretest=True,
                     bootstrap=True, boot_iterations=args.boot_iterations, seed_indexer=args.seed_indexer, outliers=False)

test,label=file.intro_sequence_evaluation()
file.solo_bootstrap(test,label)
