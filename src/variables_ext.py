import torch

# ==== gpu server ====
if torch.cuda.is_available():
    device = "cuda:0"
    n_gpu = 1
    
    # the paths for three reward functions
    clf_main_fp = r'./../../MisinfoCorrect_support_models'

    politeness_clf_fp = f"{clf_main_fp}/politeness_clf.pt"

    refutation_clf_fp = f"{clf_main_fp}/refutation_clf.pt"

    evidence_clf_fp = f"{clf_main_fp}/evidence_clf.pt"


# general text reward
if_perplexity = True
if_relevance = True # i.e., the mentioned coherence reward in the paper
# counter response reward
# False True
if_politeness = True 
if_refutation = True 
if_evidence = True 

# initially 5, 3 
every_k_epoch_save_model = 3



