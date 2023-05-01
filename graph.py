#Make essential imports
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pickle


#List of emojis, from worst to best
def set_emojis(ax):
    for i in range(10):
        path = "Emoji{}.png".format(i)
        img = plt.imread(path)
        im = OffsetImage(img, zoom=0.20)
        im.image.axes = ax
        ab = AnnotationBbox(im, (i, 0),  xybox=(0., -16.), frameon=False,
                            xycoords='data',  boxcoords="offset points", pad=26)
        ax.add_artist(ab)

#Load pre-trained parameters
with open("parameters", "rb") as f:
    loaded_dict = pickle.load(f)

#Set our w and b
w, b = loaded_dict['w'], loaded_dict['b']

#Load the dataset
df = pd.read_csv('winequality.csv')

#Get max, min, mean values of each feature for the sliders
features = [i for i in df.columns.to_list()]
del features[-1]
key_val = {}
for i in features:
    key_val.__setitem__(i, {'max': df[i].max(), 'min': df[i].min(), 'mean': df[i].mean()})

# The function to be plotted
def f(features:list()):
    input = np.array(features)
    prediction = tf.argmax(tf.nn.softmax(tf.transpose(tf.matmul(w, input.reshape(-1, 1))+tf.transpose(b))), axis=1)
    return prediction+3

#Get the mean of each feature
means = [key_val[i]['mean'] for i in key_val]

# Create the figure and the line that we will manipulate
fig, ax = plt.subplots()
scatter = ax.scatter(f(means), f(means), marker='o', color='black') #Set initial point

ax.set_yticks(range(10)) #Set number of classes as y-axis
ax.set_yticklabels(['1', '2', '3', '4', '5', '6', '7', '8', '9', '10']) #Set the correct numbers
ax.set_ylabel("Wine's quality")

ax.set_xticks(range(10)) #Set number of classes as x-axis
ax.set_xticklabels(['', '', '', '', '', '', '', '', '', '']) #Clean the axis
ax.tick_params(axis='x', which='major', pad=26)

set_emojis(ax)

# adjust the main plot to make room for the sliders
fig.subplots_adjust(left=0.25, bottom=0.25)


# Make a horizontal slider to control the fixed acidity
ax_fix_acid = fig.add_axes([0.2, 0.15, 0.3, 0.03])
fix_acid_slider = Slider(
    ax=ax_fix_acid,
    label='Fix. acidity',
    valmin=key_val['fixed acidity']['min'],
    valmax=key_val['fixed acidity']['max'],
    valinit=key_val['fixed acidity']['mean'],
)

# Make a horizontal slider to control the volatile acidity
ax_vol_acid = fig.add_axes([0.6, 0.15, 0.3, 0.03])
vol_acid_slider = Slider(
    ax=ax_vol_acid,
    label='Vol. acidity',
    valmin=key_val['volatile acidity']['min'],
    valmax=key_val['volatile acidity']['max'],
    valinit=key_val['volatile acidity']['mean'],
)

# Make a horizontal slider to control the citric acid
ax_citric_acid = fig.add_axes([0.2, 0.1, 0.3, 0.03])
citric_acid_slider = Slider(
    ax=ax_citric_acid,
    label='Cit. acid',
    valmin=key_val['citric acid']['min'],
    valmax=key_val['citric acid']['max'],
    valinit=key_val['citric acid']['mean'],
)

# Make a horizontal slider to control the residual sugar
ax_res_sugar = fig.add_axes([0.6, 0.1, 0.3, 0.03])
res_sugar_slider = Slider(
    ax=ax_res_sugar,
    label='Res. sugar',
    valmin=key_val['residual sugar']['min'],
    valmax=key_val['residual sugar']['max'],
    valinit=key_val['residual sugar']['mean'],
)

# Make a horizontal slider to control the free sulfur dioxide
ax_free_sulf_diox = fig.add_axes([0.2, 0.05, 0.3, 0.03])
free_sulf_diox_slider = Slider(
    ax=ax_free_sulf_diox,
    label='Free sulf. diox.',
    valmin=key_val['free sulfur dioxide']['min'],
    valmax=key_val['free sulfur dioxide']['max'],
    valinit=key_val['free sulfur dioxide']['mean'],
)


# Make a horizontal slider to control the chlorides
ax_chlorides = fig.add_axes([0.6, 0.05, 0.3, 0.03])
chlorides_slider = Slider(
    ax=ax_chlorides,
    label='Chlorides',
    valmin=key_val['chlorides']['min'],
    valmax=key_val['chlorides']['max'],
    valinit=key_val['chlorides']['mean'],
)

# Make a horizontal slider to control the total sulfur dioxide
ax_total_sulf_diox = fig.add_axes([0.2, 0, 0.3, 0.03])
total_sulf_diox_slider = Slider(
    ax=ax_total_sulf_diox,
    label='Total sulf. diox.',
    valmin=key_val['total sulfur dioxide']['min'],
    valmax=key_val['total sulfur dioxide']['max'],
    valinit=key_val['total sulfur dioxide']['mean'],
)

# Make a horizontal slider to control the density
ax_density = fig.add_axes([0.6, 0, 0.3, 0.03])
density_slider = Slider(
    ax=ax_density,
    label='Density',
    valmin=key_val['density']['min'],
    valmax=key_val['density']['max'],
    valinit=key_val['density']['mean'],
)

# Make a vertical slider to control the pH
ax_pH = fig.add_axes([0.02, 0.25, 0.03, 0.63])
pH_slider = Slider(
    ax=ax_pH,
    label='pH',
    valmin=key_val['pH']['min'],
    valmax=key_val['pH']['max'],
    valinit=key_val['pH']['mean'],
    orientation='vertical'
)

# Make a vertical slider to control the sulphates
ax_sulphates = fig.add_axes([0.07, 0.25, 0.03, 0.63])
sulphates_slider = Slider(
    ax=ax_sulphates,
    label='Sulphates',
    valmin=key_val['sulphates']['min'],
    valmax=key_val['sulphates']['max'],
    valinit=key_val['sulphates']['mean'],
    orientation='vertical'
)

# Make a vertical slider to control the alcohol
ax_alcohol = fig.add_axes([0.12, 0.25, 0.03, 0.63])
alcohol_slider = Slider(
    ax=ax_alcohol,
    label='Alcohol',
    valmin=key_val['alcohol']['min'],
    valmax=key_val['alcohol']['max'],
    valinit=key_val['alcohol']['mean'],
    orientation='vertical'
)

# The function to be called anytime a slider's value changes
def update(val):
    prediction = f([fix_acid_slider.val, vol_acid_slider.val, citric_acid_slider.val, res_sugar_slider.val, 
                      chlorides_slider.val, free_sulf_diox_slider.val, total_sulf_diox_slider.val, 
                      density_slider.val, pH_slider.val, sulphates_slider.val, alcohol_slider.val])
    scatter.set_offsets([prediction[0], prediction[0]]) 
    fig.canvas.draw_idle()
    print(prediction)


# Register the update function with each slider
fix_acid_slider.on_changed(update)
vol_acid_slider.on_changed(update)
citric_acid_slider.on_changed(update)
res_sugar_slider.on_changed(update)
free_sulf_diox_slider.on_changed(update)
chlorides_slider.on_changed(update)
total_sulf_diox_slider.on_changed(update)
density_slider.on_changed(update)
pH_slider.on_changed(update)
sulphates_slider.on_changed(update)
alcohol_slider.on_changed(update)

# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = fig.add_axes([0.5, 0.9, 0.1, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

#Define a reset function
def reset(event):
    fix_acid_slider.reset()
    vol_acid_slider.reset()
    citric_acid_slider.reset()
    res_sugar_slider.reset()
    free_sulf_diox_slider.reset()
    chlorides_slider.reset()
    total_sulf_diox_slider.reset()
    density_slider.reset()
    pH_slider.reset()
    sulphates_slider.reset()
    alcohol_slider.reset()

button.on_clicked(reset)

plt.show()