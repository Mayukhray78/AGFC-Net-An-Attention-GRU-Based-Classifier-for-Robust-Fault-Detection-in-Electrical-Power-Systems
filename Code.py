import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import scipy.stats as stats
df = pd.read_csv(’/content/detect_dataset.csv’)
df.head()
df.shape
df.info()
df.describe()
df.isnull().sum()
df.drop([’Unnamed: 7’,’Unnamed: 8’],axis=1,inplace = True)
df.duplicated().sum()
df[’Output (S)’].unique()
# Frequency data
value_counts = df[’Output (S)’].value_counts()
# Create subplots for bar chart and pie chart with Seaborn color palette
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
# Bar chart
ax1.bar(value_counts.index, value_counts.values, color=’salmon’, edgecolor=’black’)
ax1.set_xlabel(’Class’)
ax1.set_ylabel(’Frequency’)
ax1.set_title(’Frequency of Each Class’)
# Pie chart with Seaborn color palette
colors = sns.color_palette("husl", len(value_counts)) # You can choose other palettes ax2.pie(value_counts, autopct=’%0.2f%%’, labels=value_counts.index, colors=colors)
ax2.set_title(’Class Distribution with Seaborn Palette’)
plt.suptitle(’Frequency and Distribution of Classes’)
plt.tight_layout()
plt.show()
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr()[[’Output (S)’]].sort_values(by=’Output (S)’, ascending=False), annot=plt.title(’Correlation of Output (S) with Other Columns’)
plt.show()
ls = [’Ia’, ’Ib’, ’Ic’, ’Va’, ’Vb’, ’Vc’]
# Custom color palette for the KDE plots
colors = [’#FF6347’, ’#4682B4’, ’#32CD32’, ’#8A2BE2’, ’#FF4500’, ’#2E8B57’]
plt.figure(figsize=(12, 5))
sns.set_style("whitegrid") # Set Seaborn style for better aesthetics
for i in range(2):
for j in range(3):
plt.subplot(2, 3, i * 3 + (j + 1))
sns.kdeplot(df[ls[i * 3 + j]], color=colors[i * 3 + j], shade=True) # Added color plt.title(f’Distribution of {ls[i * 3 + j]}’) # Added title to each subplot
plt.xlabel(ls[i * 3 + j]) # Label x-axis for each plot
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.suptitle(’KDE Distribution of All Columns’, fontsize=16)
plt.show()
# Define custom color palette for each plot
palette1 = [’#FF6347’, ’#4682B4’, ’#32CD32’] # Colors for Ia, Ib, Ic
palette2 = [’#8A2BE2’, ’#FF4500’, ’#2E8B57’] # Colors for Va, Vb, Vc
# Melting the data
ls = [’Ia’,’Ib’,’Ic’,’Va’,’Vb’,’Vc’]
df_melted1 = df[ls[:3]].melt(var_name=’Variable’, value_name=’Value’)
df_melted2 = df[ls[3:]].melt(var_name=’Variable’, value_name=’Value’)
# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
# Strip plot for Ia, Ib, Ic with custom palette
sns.stripplot(x=’Variable’, y=’Value’, data=df_melted1, ax=ax1, palette=palette1)
ax1.set_title(’Strip Plot of Ia, Ib, Ic’)
# Strip plot for Va, Vb, Vc with custom palette
sns.stripplot(x=’Variable’, y=’Value’, data=df_melted2, ax=ax2, palette=palette2)
ax2.set_title(’Strip Plot of Va, Vb, Vc’)
# Set a global title and adjust layout
fig.suptitle(’Strip Plots of Different Sets of Variables’, fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to include the overall title
# Show the plot
plt.show()
# Pair plot with KDE on the diagonal and custom color palette
pair_plot = sns.pairplot(df.drop(’Output (S)’, axis=1), diag_kind=’kde’, palette=sns.color_pair_plot.fig.suptitle(’Pair Plot of Features’, fontsize=16)
pair_plot.fig.subplots_adjust(top=0.95)
plt.show()
# Custom colormap: "YlGnBu" for a vibrant look
plt.figure(figsize=(10, 6))
sns.heatmap(df.drop(’Output (S)’, axis=1).corr(), annot=True, cmap=’YlGnBu’, center=0)
plt.title(’Correlation Heatmap with YlGnBu Colormap’)
plt.show()
ls = [’Ia’, ’Ib’, ’Ic’, ’Va’, ’Vb’, ’Vc’]
plt.figure(figsize=(12, 8))
for i in range(2):
for j in range(3):
plt.subplot(2, 3, i * 3 + (j + 1))
stats.probplot(df[ls[i*3+j]], dist="norm", plot=plt)
plt.title(ls[i*3+j])
plt.tight_layout()
plt.suptitle(’QQ plots for all columns’, y=1.05, fontsize=16)
plt.show()
x_train,x_test,y_train,y_test = train_test_split(df.drop(’Output (S)’,axis=1),df[’Output # Standardize the dataset
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
# Convert back to DataFrame for PyTorch Tabular compatibility
train_data = pd.DataFrame(x_train, columns=df.columns[:-1])
train_data[’Output (S)’] = y_train.values
test_data = pd.DataFrame(x_test, columns=df.columns[:-1])
test_data[’Output (S)’] = y_test.values
# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(x_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)
# Create TensorDatasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
import torch
import torch.nn as nn
import torch.nn.functional as F
class NODE(nn.Module):
def __init__(self, input_dim):
super(NODE, self).__init__()
# Convolutional layers to extract features
self.conv1 = nn.Conv1d(input_dim, 64, kernel_size=3, padding=1)
self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
# Attention mechanism (Simple) to focus on important features
self.attn_layer = nn.Linear(128, 128)
# GRU layer to capture sequential dependencies (if any)
self.gru = nn.GRU(128, 128, batch_first=True)
# Fully connected layers
self.fc1 = nn.Linear(128, 64)
self.fc2 = nn.Linear(64, 32)
self.fc3 = nn.Linear(32, 1)
# Batch Normalization and Dropout for regularization
self.bn1 = nn.BatchNorm1d(64)
self.bn2 = nn.BatchNorm1d(128)
self.dropout = nn.Dropout(0.5)
# Activation function
self.activation = nn.LeakyReLU(negative_slope=0.1)
def forward(self, x):
# Reshape the input to [batch_size, num_channels, sequence_length]
x = x.unsqueeze(2) # Add a sequence length dimension
# Convolutional layers with Batch Normalization
x = self.activation(self.bn1(F.relu(self.conv1(x))))
x = self.activation(self.bn2(F.relu(self.conv2(x))))
# Attention mechanism
attn_weights = F.softmax(self.attn_layer(x), dim=2)
x = x * attn_weights # Apply attention weights
# GRU layer
x, _ = self.gru(x)
# Flatten the output and pass through fully connected layers
x = x[:, -1, :] # Use the last time-step
x = self.dropout(self.activation(self.fc1(x)))
x = self.dropout(self.activation(self.fc2(x)))
x = self.fc3(x)
return x
# Define the model
model = NODE(input_dim=6) # 6 input features
# Define loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Training loop
num_epochs = 50
for epoch in range(num_epochs):
model.train()
running_loss = 0.0
for inputs, labels in train_loader:
optimizer.zero_grad()
outputs = model(inputs)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
running_loss += loss.item()
print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
# Evaluation
model.eval()
with torch.no_grad():
correct = 0
total = 0
y_pred = []
y_true = []
for inputs, labels in test_loader:
outputs = model(inputs)
predicted = torch.sigmoid(outputs).round()
total += labels.size(0)
correct += (predicted == labels).sum().item()
y_pred.extend(predicted.numpy())
y_true.extend(labels.numpy())
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy:.2f}%")
