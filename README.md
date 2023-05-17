!pip install tensorflow-gpu
!nvidia-smi
# Load the Drive helper and mount
from google.colab import drive

# This will prompt for authorization.
drive.mount('/content/drive')
import os
# Directory with bacterial wilt pictures
bacterial_wilt_dir = os.path.join('/content/drive/MyDrive/training test data/training data/bacterial wilt')

# Directory with cercospora leaf spot pictures
cercospora_leaf_spot_dir = os.path.join('/content/drive/MyDrive/training test data/training data/cercospora leaf spot')

# Directory with collar_rot pictures
collar_rot_dir = os.path.join('/content/drive/MyDrive/training test data/training data/collar rot')

# Directory with healthy leaf pictures
healthy_leaf_dir = os.path.join('/content/drive/MyDrive/training test data/training data/healthy leaf')

# Directory with tobacco mosanic virus pictures
tobacco_mosanic_virus_dir = os.path.join('/content/drive/MyDrive/training test data/training data/tobacco mosanic virus')
print('total bacterial_wilt images:', len(os.listdir(bacterial_wilt_dir)))
print('total cercospora_leaf_spot images:', len(os.listdir(cercospora_leaf_spot_dir)))
print('total collar_rot images:', len(os.listdir(collar_rot_dir)))
print('total healthy_leaf images:', len(os.listdir(healthy_leaf_dir)))
print('total tobacco_mosanic_virus images:', len(os.listdir(tobacco_mosanic_virus_dir)))
train_bacterial_wilt_names = os.listdir(bacterial_wilt_dir)
print(train_bacterial_wilt_names[:5])

train_cercospora_leaf_spot_names = os.listdir(cercospora_leaf_spot_dir)
print(train_cercospora_leaf_spot_names[:5])

train_collar_rot_names = os.listdir(collar_rot_dir)
print(train_collar_rot_names[:5])

train_healthy_leaf_names = os.listdir(healthy_leaf_dir)
print(train_healthy_leaf_names[:5])

train_tobacco_mosanic_virus_names = os.listdir(tobacco_mosanic_virus_dir)
print(train_tobacco_mosanic_virus_names[:5])
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# Parameters for our graph; we'll output images in a 4x4 configuration
nrows = 4
ncols = 4

# Index for iterating over images
pic_index = 0

fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)

pic_index += 8
next_bacterial_wilt_pix = [os.path.join(bacterial_wilt_dir, fname) 
                for fname in train_bacterial_wilt_names[pic_index-8:pic_index]]
next_cercospora_leaf_spot_pix = [os.path.join(cercospora_leaf_spot_dir, fname) 
                for fname in train_cercospora_leaf_spot_names[pic_index-8:pic_index]]
next_collar_rot_pix = [os.path.join(collar_rot_dir, fname) 
                for fname in train_collar_rot_names[pic_index-8:pic_index]]
next_healthy_leaf_pix = [os.path.join(healthy_leaf_dir, fname) 
                for fname in train_healthy_leaf_names[pic_index-8:pic_index]]
next_tobacco_mosanic_virus_pix = [os.path.join(tobacco_mosanic_virus_dir, fname) 
                for fname in train_tobacco_mosanic_virus_names[pic_index-8:pic_index]]
print ("Showing some bacterial_wilt pictures...")
print()
for i, img_path in enumerate(next_bacterial_wilt_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("Showing some cercospora_leaf_spot pictures...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_cercospora_leaf_spot_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("Showing some collar_rot pictures...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_cercospora_leaf_spot_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("Showing some healthy_leaf pictures...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_healthy_leaf_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()

print ("Showing some tobacco_mosanic_virus_spot pictures...")
print()
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
for i, img_path in enumerate(next_tobacco_mosanic_virus_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()
batch_size = 128
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# All images will be rescaled by 1./255
train_datagen = ImageDataGenerator(
        rescale=1/255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Flow training images in batches of 128 using train_datagen generator
training_set = train_datagen.flow_from_directory('/content/drive/MyDrive/training test data/training data',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')
pip install -U clustimage
from clustimage import Clustimage
# Import library
from clustimage import Clustimage

# init
cl = Clustimage(method='hog', params_hog={'orientations':5, 'pixels_per_cell':(8,8)})
# cl = Clustimage(method='pca-hog')

# load example with leaves
pathnames = '/content/drive/MyDrive/training test data/training data'

# Cluster images using the input pathnames.
results = cl.fit_transform(pathnames, min_clust=4, max_clust=12)

# If you want to experiment with a different clustering and/or evaluation approach, use the cluster functionality.
# This will avoid pre-processing, and performing the feature extraction of all images again.
# You can also cluster on the 2-D embedded space by setting the cluster_space parameter 'low'
#


# Make various plots:

# Silhouette plots
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])

# PCA explained variance plot
# cl.pca.plot()

# Dendrogram
cl.dendrogram()

# Plot unique image per cluster
cl.plot_unique(img_mean=False, show_hog=True)

# Scatterplot
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)

# Plot images per cluster or all clusters
cl.plot(labels=3, show_hog=True)
# Load library
from clustimage import Clustimage
# init
cl = Clustimage(method='pca-hog')
# load example with leaves
pathnames = '/content/drive/MyDrive/training test data/training data'
# The pathnames are stored in a list
print(pathnames[0:2])


# Preprocessing, feature extraction and clustering. Lets set a minimum of 1-
results = cl.fit_transform(pathnames)

# Lets first evaluate the number of detected clusters.
# This looks pretty good because there is a high distinction between the peak for 5 clusters and the number of clusters that subsequently follow.
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])
cl.scatter(dotsize=50, zoom=None)
cl.scatter(dotsize=50, zoom=0.5)
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
cl.scatter(dotsize=50, zoom=0.5, img_mean=False)
cl.scatter(zoom=1.2, plt_all=True, figsize=(150,100))
# Plot unique images
cl.plot_unique()
cl.plot_unique(img_mean=False)

# Plot all images per cluster
cl.plot()

# Plot the images in a specific cluster
cl.plot(labels=5)
# Load library
from clustimage import Clustimage
# init
cl = Clustimage(method='pca')
# load example with leaves
pathnames = '/content/drive/MyDrive/training test data/training data'
# The pathnames are stored in a list
print(pathnames[0:2])


# Preprocessing, feature extraction and clustering. Lets set a minimum of 1-
results = cl.fit_transform(pathnames)

# Lets first evaluate the number of detected clusters.
# This looks pretty good because there is a high distinction between the peak for 5 clusters and the number of clusters that subsequently follow.
cl.clusteval.plot()
cl.clusteval.scatter(cl.results['xycoord'])
# Plot unique images
cl.plot_unique()
cl.plot_unique(img_mean=False)

# Plot all images per cluster
cl.plot()

# Plot the images in a specific cluster
cl.plot(labels=5)
# Import library
from clustimage import Clustimage

# Init with settings such as PCA
cl = Clustimage(method='hog')

# load example with leaves
pathnames = '/content/drive/MyDrive/training test data/testing data'

# Cluster leaves
results = cl.fit_transform(pathnames)

# Read the unseen image. Note that the find functionality also performs exactly the same preprocessing steps as for the clustering.
results_find = cl.find(pathnames, k=0, alpha=0.05)

# Show whatever is found. This looks pretty good.
cl.plot_find()
cl.scatter()
# Import library
from clustimage import Clustimage

# Init with settings such as PCA
cl = Clustimage(method='pca')

# load example with leaves
pathnames = '/content/drive/MyDrive/training test data/testing data'

# Cluster leaves
results = cl.fit_transform(pathnames)

# Read the unseen image. Note that the find functionality also performs exactly the same preprocessing steps as for the clustering.
results_find = cl.find(pathnames, k=0, alpha=0.05)

# Show whatever is found. This looks pretty good.
cl.plot_find()
