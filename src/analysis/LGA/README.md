README:

The calc_GDTTS.py script is used as a wrapper for the LGA package
which calculates the GDT-TS score between two PDB structures. The LGA
executable is compiled for linux systems and runs without error on Cori.

Run script by using the following command:

$python calc_GDTTS.py path_to_targetPDB.pdb path_to_decoyPDB.pdb

- pdb file paths are relative to calc_GDTTS.py file.
- the first path corresponds to the target structure
- the second path corresponds to the decoy structure

The LGA program flags are defined within the calc_GDTTS.py file in the
lga_command variable. The current flags as defined in the script will
run the GDT-TS calculation for chain 'A' on two PDBs. The length of
both chains must be the same for the calculation to work. The LGA
program output will be printed out, and results saved within the
TMP folder.

--------------------------------------------------------------------------------
To test the script, there is a test PDB in the DATA folder.

Run:

$python calc_GDTTS.py DATA/3gft.pdb DATA/3gft.pdb

Since the PDB is being used as both the target and decoy, it should produce a
GDT-TS score of 100. Near the bottom of the LGA output, you'll find line which
shows the RMSD, GDT-TS, and LGA scores:

#CA            N1   N2   DIST      N    RMSD    GDT_TS    LGA_S3     LGA_Q
SUMMARY(GDT)  167  167    4.0    167    0.00   100.000   100.000   167.000

--------------------------------------------------------------------------------
For more information about the LGA package please refer to:
http://proteinmodel.org/AS2TS/LGA/lga_format.html
