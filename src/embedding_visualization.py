from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

def draw(emb, folder, model_name ,name, label):
    folder_name = f'{model_name}_Embeddings'
    full_path = os.path.join(os.path.join('images', folder), folder_name)
    os.makedirs(full_path, exist_ok=True)

    perp = min(30, emb.shape[0]-1)
    comp = TSNE(n_components=2, perplexity=perp).fit_transform(emb)

    plt.scatter(comp[:,0], comp[:,1])
    plt.title(name)

    path = os.path.join(full_path, f'{name}_tsne.png')

    plt.savefig(path)
    plt.close()
    
    
def embedding_space_vis(model, folder, model_name, label):
    draw(model.store_embed.weight.detach().cpu().numpy(), folder, model_name, 'store embeddings', label)
    draw(model.dow_embed.weight.detach().cpu().numpy(), folder, model_name, 'day of week embeddings', label)
    draw(model.day_embed.weight.detach().cpu().numpy(), folder, model_name, 'day embeddings', label)
    draw(model.month_embed.weight.detach().cpu().numpy(), folder, model_name, 'month embeddings', label)
    draw(model.year_embed.weight.detach().cpu().numpy(), folder, model_name, 'year embeddings', label)
    draw(model.state_embed.weight.detach().cpu().numpy(), folder, model_name, 'state embeddings', label)
