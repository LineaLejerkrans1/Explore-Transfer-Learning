from fastai.vision.all import *
from sklearn.manifold import TSNE

# Load images
path = Path("/Users/linea/Kth/År4/Deep Learning/Project/DD2424_Project/oxford-iiit-pet")
files = get_image_files(path/"images")
files = L(files).shuffle() # shuffel training data

# Labeling function: Uppercase = Cat (True), Lowercase = Dog (False)
def label_func(f): return f[0].isupper()
pat = r'^(.*)_\d+.jpg'


def get_dls(path, files, pat, fraction=1.0, use_basic_aug=True):
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


    # Take a fraction of each breed
    sub_files = []
    for breed in breed_to_files:
        # Ensure at least 1 image, or based on defined fraction of each breed
        limit = max(1, int(len(breed_to_files[breed]) * fraction)) 
        sub_files.extend(breed_to_files[breed][:limit])
    
    # Shuffel the limited data array
    sub_files = L(sub_files).shuffle() 
    
    # Data Augmentation toggle 
    batch_tfms = aug_transforms(size=224) if use_basic_aug else [Normalize.from_stats(*imagenet_stats)]

    # For small datasets, reduce tthe batch size
    batch_size = 64 if fraction >= 0.1 else 8


    # Explicitly define the CategoryBlock with full vocabulary 
    # (For small datasets, since some classes might not be present in training causing valueError)
    
    # For larger data set use block toogled comment below instead
    dblock = DataBlock(
        blocks=(ImageBlock, CategoryBlock(vocab=all_breeds)),
        get_items=lambda x: x,
        get_y=using_attr(RegexLabeller(pat), 'name'),
        splitter=RandomSplitter(valid_pct=0.2, seed=42),
        item_tfms=Resize(224),
        batch_tfms=batch_tfms
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




def train(dls, cbs=None, epochs=10):
    learn = vision_learner(dls, resnet18, metrics=accuracy)
    
    # Phase 1
    print("Phase 1")
    learn.freeze()
    learn.fit_one_cycle(2, lr_max=2e-3, cbs=cbs) 
    
    # Phase 2
    print(f"Phase 2: Training whole network for {epochs} epochs")
    learn.unfreeze()
    # Use a lower learning rate 
    learn.fit_one_cycle(epochs, lr_max=slice(1e-6, 1e-4), cbs=cbs) 
    
    return learn



def run_training(path, data_pct=1.0, use_basic_aug=True, use_mixup=False, epochs=5):
    print(f"\n--- Data: {data_pct*100}% | Basic Aug: {use_basic_aug} | MixUp: {use_mixup} ---")
    
    dls = get_dls(path, files, pat, fraction=data_pct, use_basic_aug=use_basic_aug)
    
    # Add MixUp if requested
    cbs = [MixUp()] if use_mixup else []
    learn = train(dls, cbs, epochs)
    
    # Calc and print loss and accuracy 
    val_loss, val_acc = learn.validate()
    print(f"Final Validation Accuracy: {val_acc*100:.2f}%")
    
    return learn, dls, val_acc




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



def test_Run():
    results = {}
    saved_dls = {}

    tests = [0.01, 0.1]

    for pct in tests:
        results[pct] = {}
        name_str = str(pct)

        # No Augmentation Baseline
        learn_baseline, dls_baseline, acc_baseline = run_training(path, data_pct=pct, use_basic_aug=False, use_mixup=False, epochs=30)
        results[pct]['None'] = acc_baseline
        learn_baseline.export(f'netBaseline_{name_str}.pkl')

        # Basic Augmentation
        learn_basic, dls_basic, acc_basic = run_training(path, data_pct=pct, use_basic_aug=True, use_mixup=False, epochs=30)
        results[pct]['Basic'] = acc_basic
        learn_basic.export(f'netBasicAug_{name_str}.pkl')

        # Advanced Augmentation (MixUp)
        learn_mix, dls_mix, acc_mix = run_training(path, data_pct=pct, use_basic_aug=True, use_mixup=True, epochs=32)
        results[pct]['MixUp'] = acc_mix
        learn_mix.export(f'netAdvAug_{name_str}.pkl')

        # Save the advanced dataloaders
        saved_dls[pct] = dls_mix

    return




if __name__ == "__main__":

    test_Run()
    target_pct = 0.1
    target_pct_str = str(target_pct) 
    
    print(f"\nLoading models for {target_pct_str} Data Representation Analysis...")

    # Load models
    model_baseline = load_learner(f'netBaseline_{target_pct_str}.pkl', cpu=False)
    model_basic    = load_learner(f'netBasicAug_{target_pct_str}.pkl', cpu=False)
    model_mix      = load_learner(f'netAdvAug_{target_pct_str}.pkl', cpu=False)
    
    # Get dataloaders
    dls_target = get_dls(path, files, pat, fraction=target_pct, use_basic_aug=True)
    print("Generating t-SNE plots...")

    # Baseline 
    emb_no, lbls_no = get_embeddings(model_baseline, dls_target)
    plot_tsne(emb_no, lbls_no, dls_target.vocab, title=f"t-SNE: {int(target_pct*100)}% Data (No Augmentation Baseline)")

    # Basic Augmentation
    emb_basic, lbls_basic = get_embeddings(model_basic, dls_target)
    plot_tsne(emb_basic, lbls_basic, dls_target.vocab, title=f"t-SNE: {int(target_pct*100)}% Data (With Basic Augmentation)")

    # MixUp 
    emb_mix, lbls_mix = get_embeddings(model_mix, dls_target)
    plot_tsne(emb_mix, lbls_mix, dls_target.vocab, title=f"t-SNE: {int(target_pct*100)}% Data (With Advanced MixUp)")
