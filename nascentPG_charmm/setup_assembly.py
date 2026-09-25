#!/usr/bin/env python3
import sys
import os

# Usage: Usage: python setup_NGP.py <num_repeats>
if len(sys.argv) != 3:
    print("Usage: python setup_assembly.py <num_repeats> <input_dir>")
    sys.exit(1)

try:
    num_repeats = int(sys.argv[1])
except ValueError:
    sys.exit("Error: num_repeats must be an integer")

if not os.path.exists(sys.argv[2]):
    FileNotFoundError(f"The dir '{sys.argv[2]}' does not exist.")
    #sys.exit("Error: seed_pdb must be a valid file")   
else:
    input_dir = sys.argv[2]
    print(f"Using {input_dir} as a source of components")

num_sugars = num_repeats*2
num_resi = num_sugars + 1

pg_name = f"l{num_sugars}"

div_name = os.path.basename(input_dir)
exportdir = f"{div_name}_export"
if not os.path.exists(exportdir):
    os.makedirs(exportdir)
    print(f"Made directory: {exportdir}")
else:
    print(f"Directory {exportdir} exists")


print(f"Preparing str files for {pg_name}")
#tcl_path = os.path.join(REF_parentdir, f"build_{pg_name}.tcl")

seq_filename = f"custom_files/{pg_name}_seq.str"

vars_txt = f"""*Information about system
*
set DONNAME = {pg_name}
set DIVNAME = {div_name}
set PROFN = {div_name}/{div_name}_divisome
set DONFN = {div_name}/{div_name}_donor_{pg_name}
set ACCFN = {div_name}/{div_name}_acceptor_l2
set EXPORTDIR = {exportdir}
set EXPORTFN = {div_name}_divisome_{pg_name}_assembled
"""

# Write to vars.str
with open("divisome_vars.str", "w") as f:
    f.write(vars_txt)
print(f"Wrote divisome_vars.str")
print(f"Usage: charmm < assemble_docked_divisome.inp > {exportdir}/assemble_docked_divisome.out")




