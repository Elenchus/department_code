'''re-scale graphs'''
import pickle

filenames = ['hip', 'knee', 'shoulder']
graph_type = '_claims_per_episode'
models = []
maxes = []
for filename in filenames:
    model = pickle.load(open(f'{filename}{graph_type}.pickle', 'rb'))
    _, _, _, y_max = model.gca().axis()
    models.append(model)
    maxes.append(y_max)

new_y_max = max(maxes)
for i, model in enumerate(models):
    model.gca().set_ylim((0, new_y_max))
    model.savefig(f"{filenames[i]}{graph_type}.png")
