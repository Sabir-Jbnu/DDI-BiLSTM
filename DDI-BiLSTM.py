import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dropout, BatchNormalization, MaxPooling1D, Dense, LSTM, Bidirectional, Activation
from tensorflow.keras.optimizers import RMSprop
#%%
import os
# Set GPU settings
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "5"
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
#%%
# Define the model architecture
def my_model(input_shape, num_classes):
    model = Sequential([
        Conv1D(filters=16, kernel_size=3, input_shape=input_shape),
        BatchNormalization(),
        MaxPooling1D(pool_size=2, strides=2, padding='valid'),
        Dropout(0.1),
        Conv1D(filters=32, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.1),
        Conv1D(filters=48, activation='relu', kernel_size=3),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Conv1D(filters=64, activation='relu', kernel_size=3),
        BatchNormalization(),
        Bidirectional(LSTM(128, return_sequences=True)),
        Activation('relu'),
        Bidirectional(LSTM(96)),
        Dense(256, activation='relu', kernel_initializer='he_normal',
              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        Dense(units=num_classes, activation='softmax')
    ])
    return model
#%%
# Assuming input_shape and num_classes are known
input_shape = (4096, 1)
model = my_model(input_shape, num_classes=86)
#%%
# Load the saved weights
weights_path = "/home/sabir/cnn_lstm_9jan/cnn_weight.ckpt"
model.load_weights(weights_path)

# Compile the model
optimizer = RMSprop(learning_rate=0.000173)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
#%%
# Load the test data
test_data_path = "/home/sabir/cnn_lstm_9jan/test_data_LSTM.npz"
test_data = np.load(test_data_path)
X_test = test_data['X_test']
y_test = test_data['y_test']
#%%
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
#%%
# Suppress undefined metric warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import seaborn as sns

#%%
# Predict and evaluate the final trained model on the test set
y_pred_prob = model.predict(X_test)
y_true = np.argmax(y_test, axis=1)
y_pred = np.argmax(y_pred_prob, axis=1)
conf_matrix = confusion_matrix(y_true, y_pred)
#%%
n_classes = 86

# Create empty lists to store metrics for each class
class_accuracy = []
class_auc = []
class_aupr = []
class_macro_precision = []
class_macro_recall = []
class_macro_fscore = []
class_micro_precision = []
class_micro_recall = []
class_micro_fscore = []
class_mcc = []

# Calculate metrics for each class
for i in range(n_classes):
    class_accuracy.append(accuracy_score(y_true[y_true == i], y_pred[y_true == i]))
    class_auc.append(roc_auc_score((y_true == i).astype(int), y_pred_prob[:, i]))
    class_aupr.append(average_precision_score((y_true == i).astype(int), y_pred_prob[:, i]))
    
    class_precision = precision_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_recall = recall_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_fscore = f1_score((y_true == i).astype(int), (y_pred == i).astype(int))
    class_mcc.append(matthews_corrcoef((y_true == i).astype(int), (y_pred == i).astype(int)))
    
    class_macro_precision.append(class_precision)
    class_macro_recall.append(class_recall)
    class_macro_fscore.append(class_fscore)
    
    class_micro_precision.append(class_precision)
    class_micro_recall.append(class_recall)
    class_micro_fscore.append(class_fscore)

# Combine the specific metrics into a single list
metrics = [
    class_accuracy, class_macro_precision, class_macro_recall, class_macro_fscore,
    class_micro_precision, class_micro_recall, class_micro_fscore,
    class_auc, class_aupr, class_mcc
]

# Define labels for the specific metrics
metric_labels = [
    'ACC', 'Macro-Precision', 'Macro-Recall', 'Macro-F1',
    'Micro-Precision', 'Micro-Recall', 'Micro-F1',
    'AUC', 'AUPR Curve', 'MCC'
]
# Create a list of different colors for each box
box_colors = ['blue', 'green', 'red', 'purple', 'orange', 'pink', 'brown', 'cyan', 'magenta', 'lime']

# Plot a box plot for the specific metrics with different colors
plt.figure(figsize=(15, 10))
boxplot = plt.boxplot(metrics, labels=metric_labels, patch_artist=True, boxprops=dict(facecolor='white'))
for patch, color in zip(boxplot['boxes'], box_colors):
    patch.set_facecolor(color)
plt.title('Box Plot of Evaluation Metrics for all the Classes')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Metric Value')

# Custom legend for box colors
#legend_labels = [plt.Rectangle((0, 0), 1, 1, color=color) for color in box_colors]
#plt.legend(legend_labels, metric_labels)
plt.legend().set_visible(False)
plt.ylim(0.3, 1.02)
plt.show()
plt.savefig('evaluation_metrics_plot.png', dpi=400)
#%%
# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)

# Calculate precision, recall, and F1 score for each class
precision_per_class = precision_score(y_true, y_pred, average=None)
recall_per_class = recall_score(y_true, y_pred, average=None)
f1_per_class = f1_score(y_true, y_pred, average=None)
#%%
# Calculate precision, recall, and F1 score weighted by support
precision_weighted = precision_score(y_true, y_pred, average='weighted')
recall_weighted = recall_score(y_true, y_pred, average='weighted')
f1_weighted = f1_score(y_true, y_pred, average='weighted')
#%%
# Print the metrics
print("Accuracy (ACC): {:.2f}%".format(accuracy*100))
print("Weighted Precision: {:.2f}%".format(precision_weighted*100))
print("Weighted Recall: {:.2f}%".format(recall_weighted*100))
print("Weighted F1 Score: {:.2f}%".format(f1_weighted*100))
#%%
# Calculate macro-average precision, recall, and F1 score
precision_macro = precision_score(y_true, y_pred, average='macro')
recall_macro = recall_score(y_true, y_pred, average='macro')
f1_macro = f1_score(y_true, y_pred, average='macro')
#%%
print("Macro-Average Precision:{:.2f}%".format(precision_macro*100))
print("Macro-Average Recall: {:.2f}%".format(recall_macro*100))
print("Macro-Average F1 Score: {:.2f}%".format(f1_macro*100))
#%%
precision_micro = precision_score(y_true, y_pred, average='micro')
recall_micro = recall_score(y_true, y_pred, average='micro')
f1_micro = f1_score(y_true, y_pred, average='micro')
#%%
print("Micro-Average Precision: {:.2f}%".format(precision_micro*100))
print("Micro-Average Recall: {:.2f}%".format(recall_micro*100))
print("Micro-Average F1 Score: {:.2f}%".format(f1_micro*100))
#%%
# Calculate AUC and average precision (AUPR)
roc_auc_weighted = roc_auc_score(y_test, y_pred_prob, multi_class='ovr', average='weighted')
average_precision = average_precision_score(y_test, y_pred_prob, average='weighted')
# Calculate MCC (Matthews Correlation Coefficient)
mcc = matthews_corrcoef(y_true, y_pred)
#%%
print("Weighted AUC (AUC): {:.2f}%".format(roc_auc_weighted*100))
print("Average Precision (AUPR): {:.2f}%".format(average_precision*100))
print("Matthews Correlation Coefficient (MCC): {:.2f}%".format(mcc*100))
#%%
import matplotlib as plt
n_classes = 86 
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves for all classes and save to files
colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
n_plots = n_classes // 10 + (1 if n_classes % 10 else 0)
#%%
for plot_index in range(n_plots):
    plt.figure(figsize=(10, 8))
    start_class = plot_index * 10
    end_class = min(start_class + 10, n_classes)

    for i, color in zip(range(start_class, end_class), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=3,
                 label='Class {0} (AUC = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate(Specificity)', fontsize=20)
    plt.ylabel('True Positive Rate(Sensitivity)', fontsize=20)
    plt.title(f'ROC for Classes {start_class} to {end_class - 1}', fontsize=16)
    plt.legend(loc="lower right", prop={'size': 10})

    # Save each plot to a different file
    file_name = f"/home/sabir/DDI_project/DDI_end/roc_curve_plot_{plot_index + 1}.png"
    plt.savefig(file_name, dpi=500)
    plt.show()

#%%
import matplotlib.pyplot as plt
conf_matrix = confusion_matrix(y_true, y_pred)

# Normalize the confusion matrix by row (i.e., by the number of samples in each class)
cm_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

plt.figure(figsize=(40, 40))  # Create a larger figure to improve cell size

# Use a high-contrast color map and adjust linewidths and linecolor for better separation
sns.heatmap(cm_percent, cmap='afmhot_r', square=True, cbar=True,
            xticklabels=np.arange(1, n_classes + 1), yticklabels=np.arange(1, n_classes + 1),
            linewidths=1, linecolor='black')

plt.xlabel('Predicted Label', fontsize=20)
plt.ylabel('True Label', fontsize=24)
plt.title('Confusion Matrix Heatmap', fontsize=30)

plt.xticks(fontsize=16)  # Adjust tick label size and rotate for better readability
plt.yticks(fontsize=16)

plt.show()
#%%
# Create the intermediary models
last_cnn_layer_index = [index for index, layer in enumerate(model.layers) if isinstance(layer, Conv1D)][-1]
last_bilstm_layer_index = [index for index, layer in enumerate(model.layers) if isinstance(layer, Bidirectional)][-1]
final_output_layer_index = len(model.layers) - 1
#%%
model_last_cnn = model(inputs=model.input, outputs=model.layers[last_cnn_layer_index].output)
model_last_bilstm = model(inputs=model.input, outputs=model.layers[last_bilstm_layer_index].output)
model_final_output = model(inputs=model.input, outputs=model.layers[final_output_layer_index].output)
#%%
# Predict using these models
cnn_output = model_last_cnn.predict(X_test)
bilstm_output = model_last_bilstm.predict(X_test)
final_output = model_final_output.predict(X_test)
#%%
# Apply t-SNE
perplexity_value = 40  # Adjust as needed
n_iter_value = 3000    # Adjust as needed, default is usually 1000
tsne_cnn = TSNE(n_components=2,perplexity=perplexity_value, n_iter=n_iter_value, random_state=42).fit_transform(cnn_output.reshape(cnn_output.shape[0], -1))
tsne_bilstm = TSNE(n_components=2,perplexity=perplexity_value, n_iter=n_iter_value, random_state=42).fit_transform(bilstm_output)
tsne_final = TSNE(n_components=2, perplexity=perplexity_value, n_iter=n_iter_value,random_state=42).fit_transform(final_output)
#%%
# Plotting the t-SNE results
plt.figure(figsize=(36, 12))

plt.subplot(1, 3, 1)
plt.scatter(tsne_cnn[:, 0], tsne_cnn[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=10)
plt.title('t-SNE after Last CNN Layer')

plt.subplot(1, 3, 2)
plt.scatter(tsne_bilstm[:, 0], tsne_bilstm[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=10)
plt.title('t-SNE after Last BiLSTM Layer')
plt.subplot(1, 3, 3)
plt.scatter(tsne_final[:, 0], tsne_final[:, 1], c=np.argmax(y_test, axis=1), cmap='viridis', s=10)
plt.title('t-SNE after Final Layer')

plt.show()
#%%
file_name = "/home/sabir/DDI_project/DDI_end/t-SNE_visualization.png"
plt.savefig(file_name, dpi=500)
#%%
