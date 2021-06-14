from subprocess import run

cmd = f"qsub -N nbhd_dnsty -l h_vmem=80G shim.sh ldd.py"
print(cmd)
run(cmd, shell=True)
