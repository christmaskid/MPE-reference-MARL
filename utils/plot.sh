rm results/*/log_grep results/*/log_episodes

python grep_log.py results/maddpg-vanilla_direct_2agents_3000/log
python grep_log.py results/maddpg-vanilla_direct_3agents_3000/log
python grep_log.py results/maddpg-vanilla_broadcast_2agents_3000/log
python grep_log.py results/maddpg-vanilla_broadcast_3agents_3000/log

python grep_log.py results/maddpg-commloss_direct_2agents_3000/log
python grep_log.py results/maddpg-commloss_direct_3agents_3000/log
python grep_log.py results/maddpg-commloss_broadcast_2agents_3000/log
python grep_log.py results/maddpg-commloss_broadcast_3agents_3000/log

python grep_log.py results/masac-commloss_direct_2agents_3000/log
python grep_log.py results/masac-commloss_direct_3agents_3000/log
python grep_log.py results/masac-commloss_broadcast_2agents_3000/log
python grep_log.py results/masac-commloss_broadcast_3agents_3000/log

python plot_curves.py