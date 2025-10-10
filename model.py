import pandas as pd
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Pokus načtení jednoho z obrázků na základě filename
# simple_path = labels.iloc[0]["filename"]
# img = cv2.imread(simple_path)
# img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img_rgb)
# plt.title(labels.iloc[0]["mode"])
# plt.savefig("test.png")
# print("image saved")

# self-explanatory, k načtení také normalizace hodnot


class Model:
    def __init__(self, table: string): # Table: název souboru csv tabulky
        self.table = pd.read_csv(table) # Načtení tabulky
        self.filenames = tuple(self.table["filename"].values)
    

    def load_image(self, path: string):
        img = cv2.imread(path)
        img = cv2.resize(img, (64, 64)) # Zmenšení obrázku pro vyšší efektivitu
        img = img / 255.0 # RGB hodnoty jsou převedeny na formát od 0 do 1
        return img


    def get_X(self): # X: vstupové hodnoty 
        X = []
        for path in self.filenames:
            X.append(load_image(path))
        X = np.array(X) # Vytvoření matice obsahující všechny obrázky
        return X
