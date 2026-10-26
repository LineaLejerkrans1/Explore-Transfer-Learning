from fastai.vision.all import *
from fastai.callback.all import *
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import re

# Load images
path = Path("/Users/linea/Kth/År4/Deep Learning/Project/DD2424_Project/oxford-iiit-pet")
files = get_image_files(path/"images")
files = L(files).shuffle() # shuffel training data


# Labeling function: Uppercase = Cat (True), Lowercase = Dog (False)
def label_func(f): return f[0].isupper()
pat = r'^(.*)_\d+.jpg'


# Data loading function
def get_dls(path, files, pat, fraction=1.0, use_aug=True):
    # Group files by breed using the regex to ensure every class is represented

    breed_to_files = {}
    # Run though the images 
    for f in files:
        # Extract breed name
        breedName = re.search(pat, f.name)
        if breedName:
            breed = breedName.group(1)

            if breed not in breed_to_files: 
                breed_to_files[breed] = []
            
            # Add image to right breed key in dictunariy
            breed_to_files[breed].append(f)

    all_breeds = sorted(list(breed_to_files.keys()))

    # Select fraction of each breed
    sub_files = []
    for breed in breed_to_files:
        # Ensure at least 1 image, or based on defined fraction of each breed
        limit = max(1, int(len(breed_to_files[breed]) * fraction)) 
        sub_files.extend(breed_to_files[breed][:limit])
    
    # Shuffel the limited data array
    sub_files = L(sub_files).shuffle() 
    
    # Data Augmentation toggle 
    batch_tfms = aug_transforms(size=224) if use_aug else None

    # For small datasets, reduce tthe batch size
    batch_size = 64 if fraction > 0.1 else 8

    # Explicitly define the CategoryBlock with full vocabulary 
    # (For small datasets, since some classes might not be present in training causing valueError)
    
    # For larger data set use block toogled comment below instead
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=all_breeds)),
        get_items=lambda x: x,
        get_y=using_attr(RegexLabeller(pat), 'name'),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(448),
        batch_tfms=aug_transforms(size=224) if use_aug else None
    )
    return dblock.dataloaders(sub_files, bs=batch_size)

    """ return ImageDataLoaders.from_name_re(
        path, sub_files, pat, 
        item_tfms=Resize(448),
        batch_tfms=batch_tfms,
        valid_pct=0.2, 
        seed=42,
        bs=batch_size,
        blocks=(ImageBlock, CategoryBlock(vocab=all_breeds))
    ) """




# Strategy 2, Gradualy unfrezing the net
def train_strategy_gradual(dls, wd=0.01):
    learn = vision_learner(dls, resnet18, metrics=accuracy, wd=wd)

    for l in range(1,4):
        print(l)
        learn.freeze_to(-l)
        # Find a new learning rate for each level of unfreezing
        suggestion =learn.lr_find()
        lr = suggestion.valley 
        #learn.fit_one_cycle(2, lr_max = lr)

        learn.fit_one_cycle(1, lr_max = lr)

    return learn


# Fetch embeddings from the pnemult layer 
def get_embeddings(learn, dls):
    # dentify the device, to avoid miss-match errors 
    device = dls.device
    
    # Get body and pooling layer 
    encoder = learn.model[0] 
    pooling = learn.model[1][:-1] 
    
    # Combine
    feature_extractor = nn.Sequential(encoder, pooling).to(device).eval()
    
    embeddings, labels = [], []
    with torch.no_grad():
        for x, y in dls.valid:
            # feture from feature_extractor to embeddings
            # to avoid error with numpy 
            feats = feature_extractor(x.to(device))
            embeddings.append(feats.detach().cpu().numpy())
            labels.extend(y.cpu().numpy())
            
    # Flatten the embeddings 
    return np.concatenate(embeddings).reshape(len(labels), -1), np.array(labels)




# Draws the t-sne graph to visulize the diffrent breed clusters
def plot_tsne(embeddings, labels, vocab, title="t-SNE"):
    # Create t-sne model that can reduce any data to 2D
    tsne = TSNE(n_components=2, random_state=42)  
    # Finds a way to map higher dimentions of the embedding onto 2D
    emb_2d = tsne.fit_transform(embeddings)
    
    # To visulize the t-sne graph
    plt.figure(figsize=(10, 8))

    # List names of the breeds to plot
    chosen_breeds = ['Bengal', 'beagle', 'boxer', 'Siamese', 'pug', 
                    'Sphynx', 'saint_Bernard', 'Russian_Blue', 'Maine_Coon', 'samoyed']

    # Run trough the breeds and plot them
    for breed_name in chosen_breeds:
        if breed_name in vocab:
        # Find the index number for the breed name
            i = vocab.o2i[breed_name] 
            idx = labels == i
            plt.scatter(emb_2d[idx, 0], emb_2d[idx, 1], label=breed_name, alpha=0.7, s=30)
            
    plt.title(title, fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()


# Shows the k-nearest images of a choosen image
def show_knn(learn, dls, image_index=0, k=5):
    #embeddings, labels = get_embeddings(learn, dls)
    embeddings, labels = get_embeddings(learn, dls)
    
    query_emb = embeddings[image_index].reshape(1, -1)
    query_label = dls.vocab[labels[image_index]]
    
    # Create a knn object 
    knn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
    # Run the featurs
    knn.fit(embeddings)
    # Get neighbours 
    distances, indices = knn.kneighbors(query_emb)
    neighbor_indices = indices[0][1:] 
    
    # Create subplots
    fig, axes = plt.subplots(1, k+1, figsize=(15, 3))
    
    dls.valid.dataset[image_index][0].show(ctx=axes[0])
    axes[0].set_title(f"Search Images\nBreed: {query_label}")
    axes[0].axis('off')
    
    # Run though the neighbours 
    for i, idx in enumerate(neighbor_indices):
        neighbor_label = dls.vocab[labels[idx]]
        dls.valid.dataset[idx][0].show(ctx=axes[i+1])

        # The neighbor is the same breed as search image, the name is green otherwise red
        color = 'green' if neighbor_label == query_label else 'red'
        axes[i+1].set_title(f"Neighbor {i+1}\nBreed: {neighbor_label}", color=color)
        axes[i+1].axis('off')
        
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Prepare Data
    dls_eval = get_dls(path, files, pat, fraction=1.0, use_aug=False)

    # Test the loaded model without fine tuning
    print("Pre-trained Model Analysis")
    learn_untrained = vision_learner(dls_eval, resnet18)
    emb_before, labels_before = get_embeddings(learn_untrained, dls_eval)
    plot_tsne(emb_before, labels_before, dls_eval.vocab, "t-SNE: Before Fine-Tuning")

    print("\nTraining Model")
    #learnGrad = train_strategy_gradual(dls_eval)

    # Save model to disk
    #learnGrad.export('my_pet_model.pkl') 
    print("Model saved to file")

    # Test the trained model
    print("\nTrained Model Analysis")

    # Load model from disk if present 
    learnGrad = load_learner('my_pet_model.pkl') # load 
    emb_after, labels_after = get_embeddings(learnGrad, dls_eval)
    plot_tsne(emb_after, labels_after, dls_eval.vocab, "t-SNE: After Fine-Tuning")

    print("\nShow nearest neighbours")
    #show_knn(learnGrad, dls_eval, image_index=42, k=5)
