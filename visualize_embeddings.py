
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Optional: Import UMAP if you decide to use it later
# import umap

def main():
    assert len(sys.argv) >= 4, sys.argv

    train_file = sys.argv[1]
    visualize_file = sys.argv[2]
    output_file = sys.argv[3]
    use_umap = bool(int(sys.argv[4])) if len(sys.argv) > 4 else False
    num_icl_examples = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    manual_limits = None
    if len(sys.argv) >= 10:
        try:
            manual_limits = {
                'xmin': float(sys.argv[6]),
                'xmax': float(sys.argv[7]),
                'ymin': float(sys.argv[8]),
                'ymax': float(sys.argv[9])
            }
            print(f"Manual zoom limits detected: X({manual_limits['xmin']}, {manual_limits['xmax']}) Y({manual_limits['ymin']}, {manual_limits['ymax']})")
        except ValueError:
            print("Error: Zoom limits must be numeric.")
            sys.exit(1)

    if use_umap:
        import umap

    # 2. Load Data
    print(f"Loading training data (Map) from {train_file}...")
    try:
        X_train = np.load(train_file)
        X_train_num = X_train.shape[0]
        X_vis = np.load(visualize_file)
        X_vis_num = X_vis.shape[0]
    except Exception as e:
        print(f"Error loading numpy files: {e}")
        sys.exit(1)

    # Basic shape checks
    assert X_train.ndim == 2, X_train.shape
    assert X_vis.ndim == 2, X_vis.shape

    # Split by ICL example
    if num_icl_examples > 0:
        print(f"num_icl_examples = {num_icl_examples} > 0 will work if all episodes had the same number of steps (i.e., no EoS action)")

        assert X_vis.shape[0] % num_icl_examples == 0, f"{X_vis.shape} vs {num_icl_examples}"

        all_X_vis = [[] for _ in range(num_icl_examples)]
        all_X_vis_num = []

        for i in range(X_vis.shape[0]):
            idx = i % num_icl_examples
            all_X_vis[idx].append(X_vis[i])

        for i in range(len(all_X_vis)):
            all_X_vis[i] = np.array(all_X_vis[i])

            assert all_X_vis[i].ndim == 2, all_X_vis[i].ndim

            all_X_vis_num.append(all_X_vis[i].shape[0])
    else:
        all_X_vis = [X_vis]
        all_X_vis_num = [X_vis_num]

    assert len(all_X_vis) == len(all_X_vis_num)

    print(f"  Training Data Shape: {X_train.shape}")
    print(f"  Visualize Data Shape: {X_vis.shape}")

    if X_train.shape[1] != X_vis.shape[1]:
        print(f"Error: Dimension mismatch! Train has {X_train.shape[1]} dims, but Vis has {X_vis.shape[1]} dims.")
        sys.exit(1)

    # 3. Dimensionality Reduction
    if not use_umap:
        # --- OPTION A: PCA (Standard, No extra deps) ---
        reducer = PCA(n_components=2)
        method = "PCA"
    else:
        # --- OPTION B: UMAP (Better for clustering, requires 'pip install umap-learn') ---
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            #metric='cosine',
            metric='euclidean',
        )
        method = "UMAP"

    print(f"Fitting {method} model on training data...")

    # Fit ONLY on the training data (The "Map")
    reducer.fit(X_train)

    # Transform both datasets into the same 2D coordinate system
    print("Transforming data to 2D...")
    train_2d = reducer.transform(X_train)

    # 4. Plotting
    print("Generating plot...")
    plt.figure(figsize=(12, 10), dpi=100)

    # Plot the "Map" (Background / Valid Actions)
    # Using alpha=0.3 so we can see density, and a neutral color (gray/blue)
    plt.scatter(
        train_2d[:, 0], train_2d[:, 1],
        c='slategray', alpha=0.4, s=30, label=f'Valid Embeddings (The Map): {X_train_num}',
        edgecolors='none'
    )

    # Plot the "Traveler" (Foreground / Actor Output)
    # Using alpha=0.8 and a bright color (red) to highlight where the actor is pointing
    for idx, (_X_vis, _X_vis_num) in enumerate(zip(all_X_vis, all_X_vis_num), 1):
        vis_2d = reducer.transform(_X_vis)
        plt.scatter(
            vis_2d[:, 0], vis_2d[:, 1],
            #c='crimson',
            c=f"C{idx - 1}",
            alpha=0.7, s=40, label=f'Proto-Actions (The Actor; example #{idx}): {_X_vis_num}',
            edgecolors='black', linewidth=0.5
        )

    # Styling
    plt.title(f"Embedding Space Projection ({method})\nFeature Dim: {X_train.shape[1]} -> 2", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # 5. Save Output
    # --- SAVE 1: Full View ---
    print(f"Saving full plot to {output_file} ...")
    plt.savefig(output_file)

    # --- SAVE 2: Manual Zoom-in ---
    if manual_limits:
        plt.xlim(manual_limits['xmin'], manual_limits['xmax'])
        plt.ylim(manual_limits['ymin'], manual_limits['ymax'])

        zoom_output = f"{output_file}.zoom-in.png"
        print(f"Saving manual zoom to {zoom_output} ...")
        plt.savefig(zoom_output)
    else:
        print("No manual limits provided. Skipping zoomed plot.")

    print("Done!")

if __name__ == "__main__":
    main()
