import glob
import heapq
import os
import pandas as pd
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from multiprocessing import Pool, cpu_count, set_start_method
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
import random
import argparse
from scipy.spatial import distance
import matplotlib.patheffects as patheffects

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--images_pth", help="path of images", required=True)
    parser.add_argument("-o", "--output_dir", help="path of the output directory", required=True)
    parser.add_argument("-csv", "--output_file", help="name of the output csv file without extension", required=True)
    parser.add_argument("-n", "--num_labels", help="number of labels to generate", default=300, type=int)
    return parser.parse_args()

def show_anns(anns, img):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=False)
    for ann in sorted_anns:
        print(f"Area: {ann['area']}")
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('white')  # Set the figure background color to white
        ax.set_facecolor('white')  # Set the axes background color to white
        m = ann['segmentation']
        # Set the color to yellow (R=1, G=1, B=0, A=1)
        color_mask = np.array([1, 1, 0, 1])
        # Create a black background only on the size of the image
        img_mask = np.zeros((img.shape[0], img.shape[1], 4))
        img_mask[m] = color_mask
        # Create a white canvas larger than the image
        canvas = np.ones((img.shape[0], img.shape[1], 4))
        canvas[:, :, :3] = 0  # Set RGB to white
        canvas[:, :, 3] = 1  # Set alpha to 1 for opacity
        # Overlay the black background and yellow mask on the white canvas
        canvas[img_mask[:, :, 3] > 0] = img_mask[img_mask[:, :, 3] > 0]
        ax.imshow(np.ones_like(img) * 255)  # Display the original image as the background
        ax.imshow(canvas)  # Overlay the mask with some transparency
        ax.axis('off')
        plt.show()

def calculate_centroid(mask):
    indices = np.argwhere(mask)
    if len(indices) == 0:
        return None
    centroid = np.mean(indices, axis=0).astype(int)
    return tuple(centroid)

def is_far_enough(point, selected_points, min_distance):
    for selected_point in selected_points:
        if distance.euclidean(point, selected_point) < min_distance:
            return False
    return True

def generate_smart_points(mask_generator, img, num_labels, radius=10, grid_size=10, min_distance=20):
    masks, _ = mask_generator.generate(img)
    masked_image = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    sorted_masks = sorted(masks, key=lambda x: x["area"], reverse=True)
    # print(f"Number of masks: {len(sorted_masks)}")

    centroids = []
    for mask in sorted_masks:
        m = mask['segmentation']
        centroid = calculate_centroid(m)
        if centroid is not None:
            centroids.append(centroid)
            masked_image[m] = 1  # Mark as filled

    selected_points = centroids[:num_labels]

    grid_height, grid_width = img.shape[0] // grid_size, img.shape[1] // grid_size

    # Track the number of points in each cell using a priority queue
    cell_point_counts = {(i, j): 0 for i in range(grid_size) for j in range(grid_size)}
    for (x, y) in selected_points:
        cell_i, cell_j = x // grid_height, y // grid_width
        if (cell_i, cell_j) not in cell_point_counts:
            cell_point_counts[(cell_i, cell_j)] = 0
        cell_point_counts[(cell_i, cell_j)] += 1

    # Initialize the priority queue (min-heap) with the initial counts
    pq = [(count, (i, j)) for (i, j), count in cell_point_counts.items()]
    heapq.heapify(pq)

    continue_ = False

    # Ensure even distribution by filling grid cells
    initial_points_count = len(selected_points)
    while len(selected_points) < num_labels:
        if not pq:
            print("Heap is empty but the required number of labels has not been reached.")
            break  # Break out of the loop if the heap is empty

        count, (i, j) = heapq.heappop(pq)
        if count == 0 or len(selected_points) < num_labels:
            # Find a point in this cell
            cell_indices = np.argwhere(
                (i * grid_height <= np.arange(masked_image.shape[0])[:, None]) & 
                (np.arange(masked_image.shape[0])[:, None] < (i + 1) * grid_height) & 
                (j * grid_width <= np.arange(masked_image.shape[1])) & 
                (np.arange(masked_image.shape[1]) < (j + 1) * grid_width)
            )
            
            if len(cell_indices) > 0:
                max_attempts = 100  # Maximum number of attempts to find a valid point
                attempts = 0
                best_point = None
                min_distance = 10  # Minimum distance from already selected points
                while attempts < max_attempts:
                    random_point = tuple(cell_indices[random.randint(0, len(cell_indices) - 1)])
                    
                    # Check if the random point is far from already selected points
                    distances = [np.linalg.norm(np.array(random_point) - np.array(p)) for p, _ in selected_points]
                    if all(d > min_distance for d in distances):
                        best_point = random_point
                        break
                    attempts += 1

                if best_point is None:
                    # If no valid point is found, select a random point
                    random_index = random.randint(0, len(cell_indices) - 1)
                    random_point = tuple(cell_indices[random_index])
                    best_point = random_point
                else:
                    # print(f"Selected best point {best_point[0]} with majority percentage {best_majority_percentage:.2f} in cell ({i}, {j}) after {max_attempts} attempts.")
                    random_point = best_point

                selected_points.append(random_point)
                cell_point_counts[(i, j)] += 1  # Update the count for the cell
                # Re-add the cell to the heap with the updated count
                heapq.heappush(pq, (cell_point_counts[(i, j)], (i, j)))
                # print(f"Added point {random_point} with label {majority_label} in cell ({i}, {j})")

    # Ensure the number of points does not exceed num_labels
    if len(selected_points) > num_labels:
        selected_points = selected_points[:num_labels]

    # print(f"Total selected points: {len(selected_points_and_labels)}")

    return selected_points

def show_points(coords, ax, marker_size=30, marker_color='blue', edge_color='white'):
    coords = np.array(coords)  # Convert to numpy array
    ax.scatter(coords[:, 1], coords[:, 0], color=marker_color, marker='*', s=marker_size, edgecolor=edge_color, linewidth=1.25)
    numbers = range(2, len(coords) + 2)  # Start numbering from 2
    sorted_indices = np.argsort(coords[:, 0])  # Sort by the y-coordinate (row)
    for i, idx in enumerate(sorted_indices):
        y, x = coords[idx]
        ax.text(x, y, str(numbers[i]), color='white', fontsize=12, ha='center', va='center', 
                path_effects=[patheffects.withStroke(linewidth=3, foreground='black')])

def process_image(mask_generator, img_filename, num_labels, output_dir):
    data = []

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    # Read the ground truth image
    img = cv2.imread(img_filename, cv2.COLOR_BGR2RGB)

    points = generate_smart_points(mask_generator, img, num_labels)

    # Get the extension from the img_filename
    _, ext = os.path.splitext(img_filename)

    for (pos_i, pos_j) in points:
        image_name = os.path.basename(img_filename)
        image_name = os.path.splitext(image_name)[0] + ext
        data.append([image_name, pos_i, pos_j, '-'])

    # Sort data by the 'Row' (pos_i) column
    data.sort(key=lambda x: x[1])

    # Create a black image (RGBA format)
    black = img.copy()
    black = black.astype(float) / 255
    black = np.dstack((black, np.ones((black.shape[0], black.shape[1]))))  # Add alpha channel

    # Image dimensions (1024x768)
    height, width, _ = black.shape

    dpi = 100  # Dots per inch (or use 200 or higher for finer quality)

    # Set the figure size to match the image size (1024x768) in pixels
    figsize = (width / dpi, height / dpi)  # Use the exact image dimensions in inches

    # Prepare output directory
    if output_dir[-1] != '/':
        output_dir += '/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate marker size proportional to image size
    marker_size = (height * width) / 20000 # Adjust the divisor to control the marker size

    # Save sparse points image
    plt.figure(figsize=figsize, dpi=dpi)  # Use the exact figure size
    black_uint8 = (black * 255).astype(np.uint8)  # Convert to uint8
    plt.imshow(cv2.cvtColor(black_uint8, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for displaying
    show_points(points, plt.gca(), marker_size, marker_color='black', edge_color='yellow')
    plt.gca().set_aspect('equal', adjustable='box')  # Ensure the aspect ratio is 1:1
    plt.xlim(0, width)  # Set x-axis limit to image width
    plt.ylim(height, 0)  # Set y-axis limit to image height (reverse y-axis)
    plt.axis('off')  # Hide the axes
    plt.tight_layout(pad=0)  # Adjust layout to fit the image within bounds
    plt.savefig(output_dir + image_name + '_sparse.png', dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()

    return data

def process_images(images_pth, output_dir, output_file, num_labels=300):
    data = []

    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Initialize the model and mask generator inside the process
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

    sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
    sam2_model.to(device)
    mask_generator = SAM2AutomaticMaskGenerator(model=sam2_model,
                                               points_per_side=64,
                                               points_per_patch=128,
                                               pred_iou_threshold=0.7,
                                               stability_score_thresh=0.92,
                                               stability_score_offset=0.7,
                                               crop_n_layers=1,
                                               box_nms_thresh=0.7,
                                               )

    # Get the list of images from images_pth
    if os.path.isfile(images_pth):
        image_files = [images_pth]
    else:
        image_files = glob.glob(images_pth + '/*.*')

    # Ensure num_labels is an integer
    num_labels = int(num_labels)

    for img in tqdm(image_files, desc="Processing images"):
        result = process_image(mask_generator, img, num_labels, output_dir)
        data.extend(result)

    # Modify the output filename to include the number of labels and add .csv extension
    modified_output_file = f"{output_file}_{num_labels}.csv"

    output_df = pd.DataFrame(data, columns=['Name', 'Row', 'Column', 'Label'])
    output_df.to_csv(modified_output_file, index=False)

if __name__ == "__main__":
    set_start_method('spawn')
    args = parse_arguments()
    process_images(args.images_pth, args.output_dir, args.output_file, args.num_labels)