python tools/test_canary.py --miner-type cpu --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 --seed 42 -o full_cpu_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz  2>&1 | tee full_cpu_30m.log
python tools/test_canary.py --miner-type metal --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 --seed 42 -o full_metal_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz  2>&1 | tee full_metal_30m.log
python tools/test_canary.py --miner-type qpu --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 --seed 42 -o full_qpu_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz  2>&1 | tee full_qpu_30m.log
python tools/test_canary.py --miner-type cuda --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 --seed 42 -o full_cuda_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz  2>&1 | tee full_cuda_30m.log

python tools/compare_mining_rates.py --miner-type cpu --difficulty-energy -3520 --duration 30m --min-diversity 0.1 --min-solutions 5 -o z9_t2_cpu_30m.json  2>&1 | tee z9_t2_cpu_30m.log
python tools/compare_mining_rates.py --miner-type metal --difficulty-energy -3520 --duration 30m --min-diversity 0.1 --min-solutions 5 -o z9_t2_metal_30m.json  2>&1 | tee z9_t2_metal_30m.log
python tools/compare_mining_rates.py --miner-type qpu --difficulty-energy -3520 --duration 30m --min-diversity 0.1 --min-solutions 5 -o z9_t2_qpu_30m.json  2>&1 | tee z9_t2_qpu_30m.log
python tools/compare_mining_rates.py --miner-type cuda --difficulty-energy -3520 --duration 30m --min-diversity 0.1 --min-solutions 5 -o z9_t2_cuda_30m.json  2>&1 | tee z9_t2_cuda_30m.log
python ../tools/visualize_mining_performance.py --threshold-min -3560 --threshold-max -3400 --threshold-step 10 *.log 

python tools/compare_mining_rates.py --miner-type cpu --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 -o full_cpu_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz 2>&1 | tee full_cpu_30m.log
python tools/compare_mining_rates.py --miner-type metal --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 -o full_metal_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz 2>&1 | tee full_metal_30m.log
python tools/compare_mining_rates.py --miner-type qpu --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 -o full_qpu_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz 2>&1 | tee full_qpu_30m.log
python tools/compare_mining_rates.py --miner-type cuda --difficulty-energy -14900 --duration 30m --min-diversity 0.1 --min-solutions 5 -o full_cuda_30m.json --topology dwave_topologies/topologies/advantage2_system1_8.json.gz 2>&1 | tee full_cuda_30m.log
python ../tools/visualize_mining_performance.py --threshold-min -15000 --threshold-max -14700 --threshold-step 10 --topology ../dwave_topologies/topologies/advantage2_system1_8.json.gz *.log

