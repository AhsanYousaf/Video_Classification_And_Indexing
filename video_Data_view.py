import os
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img


train_dir='Input/train/'
val_dir='Input/val/'


def count_exp(path, set_):
    dict_ = {}
    for expression in os.listdir(path):
        dir_ = path + expression
        dict_[expression] = len(os.listdir(dir_))
    df = pd.DataFrame(dict_, index=[set_])
    return df
train_count = count_exp(train_dir, 'train')
val_count = count_exp(val_dir, 'Validation')
print(train_count)
print(val_count)

train_count.transpose().plot(kind='bar')
plt.savefig('dataset_view/train_count.png')
val_count.transpose().plot(kind='bar')
plt.savefig('dataset_view/val_count.png')



plt.figure(figsize=(14,22))
i = 1
for expression in os.listdir(train_dir):
    img = load_img((train_dir + expression +'/'+ os.listdir(train_dir + expression)[1]))
    plt.subplot(1,10,i)
    plt.imshow(img)
    plt.title(expression)
    plt.axis('off')
    i += 1
plt.savefig('dataset_view/data_view.png')
plt.show()