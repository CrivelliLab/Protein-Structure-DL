## Generate Plots from training Results

# NEW GCNN
python3 src/analysis/plots.py -v --loss --acc --prec --recall --f1 --auc out/krashras_graph_new/
python3 src/analysis/plots.py -v --loss --acc --prec --recall --f1 --auc out/psiblast_graph_new/
python3 src/analysis/plots.py -v --loss --acc --prec --recall --f1 --auc out/kinase_graph_new/
python3 src/analysis/plots.py -v --loss --acc --prec --recall --f1 --auc out/oncogenes_graph_new/
python3 src/analysis/plots.py -v --loss --acc --prec --recall --f1 out/enzyme_graph_new/
