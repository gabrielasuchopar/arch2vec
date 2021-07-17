import json
import matplotlib.pyplot as plt

def plot_over_time_comparison(cmap=plt.get_cmap("tab10")):
    fig = plt.figure()
    ax_test = fig.add_subplot(1, 1, 1)

    f_reinforce_search_supervised = open('../arch2vec/models/pretrained/dim-16/reinforce-runs/run_1_arch2vec_orig_training.json')
    f_bo_search_arch2vec = open('../../info-nas/data/vae_checkpoints/2021-17-07_17-14-47/dngo-runs/run_1_features_model_orig_epoch-29.json')
    f_reinforce_search_arch2vec1 = open('../../info-nas/data/vae_checkpoints/2021-17-07_17-14-47/reinforce-runs/run_1_features_reinforce_model_orig_epoch-29.json')
    f_reinforce_search_arch2vec2 = open('../../info-nas/data/vae_checkpoints/2021-17-07_17-14-47/reinforce-runs/run_2_features_reinforce_model_orig_epoch-29.json')
    f_reinforce_search_arch2vec3 = open('../../info-nas/data/vae_checkpoints/2021-17-07_17-14-47/reinforce-runs/run_3_features_reinforce_model_orig_epoch-29.json')
    results_reinforce_search_arch2vec1 = json.load(f_reinforce_search_arch2vec1)
    results_reinforce_search_arch2vec2 = json.load(f_reinforce_search_arch2vec2)
    results_reinforce_search_arch2vec3 = json.load(f_reinforce_search_arch2vec3)
    results_bo_search_arch2vec = json.load(f_bo_search_arch2vec)
    results_reinforce_search_supervised = json.load(f_reinforce_search_supervised)
    f_reinforce_search_arch2vec1.close()
    f_reinforce_search_arch2vec2.close()
    f_reinforce_search_arch2vec3.close()
    f_bo_search_arch2vec.close()
    f_reinforce_search_supervised.close()

    ax_test.plot(results_reinforce_search_supervised['runtime'], results_reinforce_search_supervised['regret_test'], linestyle='--', marker='.', markevery=1e3, color=cmap(7), lw=2, markersize=4, label='{}: {}'.format('original arch2vec', 'REINFORCE'))
    ax_test.plot(results_reinforce_search_arch2vec1['runtime'], results_reinforce_search_arch2vec1['regret_test'], linestyle='-.',  marker='.', markevery=1e3, color=cmap(0), lw=2, markersize=4, label='{}: {}'.format('extended arch2vec (run 1)', 'REINFORCE'))
    ax_test.plot(results_reinforce_search_arch2vec2['runtime'], results_reinforce_search_arch2vec2['regret_test'], linestyle='-.',  marker='.', markevery=1e3, color=cmap(1), lw=2, markersize=4, label='{}: {}'.format('extended arch2vec (run 2)', 'REINFORCE'))
    ax_test.plot(results_reinforce_search_arch2vec3['runtime'], results_reinforce_search_arch2vec3['regret_test'], linestyle='-.',  marker='.', markevery=1e3, color=cmap(2), lw=2, markersize=4, label='{}: {}'.format('extended arch2vec (run 3)', 'REINFORCE'))
    ax_test.plot(results_bo_search_arch2vec['runtime'], results_bo_search_arch2vec['regret_test'], linestyle='-.',  marker='v', markevery=1e3, color=cmap(3), lw=2, markersize=4, label='{}: {}'.format('extended arch2vec', 'DNGO'))

    ax_test.set_xscale('log')
    ax_test.set_yscale('log')
    ax_test.set_xlabel('estimated wall-clock time [s]', fontsize=12)
    ax_test.set_ylabel('test regret', fontsize=12)
    ax_test.legend(prop={"size":10})

    plt.show()

if __name__ == '__main__':
    plot_over_time_comparison()
