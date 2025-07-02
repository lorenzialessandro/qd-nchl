python3 plot_pyribs_optimizer.py expruns/path/CartPole-v1.txt plot/C/pdf "[4, 4, 2]"
python3 trajectories_pyribs.py expruns/path/CartPole-v1.txt "[4, 4, 2]" "CartPole-v1" pdf/C/T --pdf
python3 plot_pyribs_optimizer.py expruns/path/MountainCar-v0.txt plot/M/pdf "[2, 8, 3]"
python3 trajectories_pyribs.py expruns/path/MountainCar-v0.txt "[2, 8, 3]" "MountainCar-v0" pdf/M/T --pdf
python3 plot_pyribs_optimizer.py expruns/path/LunarLander-v3.txt plot/L/pdf "[8, 8, 4]"
python3 trajectories_pyribs.py expruns/path/LunarLander-v3.txt "[8, 8, 4]" "LunarLander-v3" pdf/L/T --pdf