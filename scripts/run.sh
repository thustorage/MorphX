# python sweep_pd.py --mode pd-offline --workload baseline
# python sweep_pd.py --mode pd-offline --workload chunked
# python sweep_pd.py --mode pd-offline --workload stream
# python sweep_pd.py --mode pd-offline --workload smsched
# python sweep_pd.py --mode pd-offline --workload all --split-hint 88
# python sweep_pd.py --mode pd-offline --workload all --split-hint 76
python sweep_pd.py --mode pd-online --workload all --split-hint 88
python sweep_pd.py --mode pd-online --workload all --split-hint 76