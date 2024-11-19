import numpy as np
import matplotlib.pyplot as plt
import copy
import random

# Assuming keyfacial_df is already defined from the previous code

def visualize_transformations(keyfacial_df):
    # Create a figure with subplots for different transformations
    fig, axs = plt.subplots(2, 2, figsize=(16, 16))
    fig.suptitle('Image Transformations', fontsize=16)

    # Original Image
    axs[0, 0].imshow(keyfacial_df['Image'][0], cmap='gray')
    axs[0, 0].set_title('Original Image')
    for j in range(1, 31, 2):
        axs[0, 0].plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')

    # Horizontal Flip
    keyfacial_df_copy = copy.copy(keyfacial_df)
    keyfacial_df_copy['Image'] = keyfacial_df_copy['Image'].apply(lambda x: np.flip(x, axis=1))
    
    # Update x coordinates for horizontal flip
    columns = keyfacial_df_copy.columns[:-1]
    for i in range(len(columns)):
        if i % 2 == 0:
            keyfacial_df_copy[columns[i]] = keyfacial_df_copy[columns[i]].apply(lambda x: 96. - float(x))

    axs[0, 1].imshow(keyfacial_df_copy['Image'][0], cmap='gray')
    axs[0, 1].set_title('Horizontal Flip')
    for j in range(1, 31, 2):
        axs[0, 1].plot(keyfacial_df_copy.loc[0][j-1], keyfacial_df_copy.loc[0][j], 'rx')

    # Brightness Increase
    keyfacial_df_brightness = copy.copy(keyfacial_df)
    keyfacial_df_brightness['Image'] = keyfacial_df_brightness['Image'].apply(
        lambda x: np.clip(random.uniform(1.5, 2) * x, 0.0, 255)
    )

    axs[1, 0].imshow(keyfacial_df_brightness['Image'][0], cmap='gray')
    axs[1, 0].set_title('Brightness Increased')
    for j in range(1, 31, 2):
        axs[1, 0].plot(keyfacial_df.loc[0][j-1], keyfacial_df.loc[0][j], 'rx')

    # Vertical Flip
    keyfacial_df_vertical = copy.copy(keyfacial_df)
    keyfacial_df_vertical['Image'] = keyfacial_df_vertical['Image'].apply(lambda x: np.flip(x, axis=0))

    # Update y coordinates for vertical flip
    for i in range(len(columns)):
        if i % 2 == 1:
            keyfacial_df_vertical[columns[i]] = keyfacial_df_vertical[columns[i]].apply(lambda x: 96. - float(x))

    axs[1, 1].imshow(keyfacial_df_vertical['Image'][0], cmap='gray')
    axs[1, 1].set_title('Vertical Flip')
    for j in range(1, 31, 2):
        axs[1, 1].plot(keyfacial_df_vertical.loc[0][j-1], keyfacial_df_vertical.loc[0][j], 'rx')

    # Adjust layout and display
    plt.tight_layout()
    plt.show()

# Call the visualization function
visualize_transformations(keyfacial_df)
