# File is for making dataset of different photos of product. Products of the same type was made in different quality
from pypylon import pylon
import pandas as pd

img = pylon.PylonImage()
tlf = pylon.TlFactory.GetInstance()

cam = pylon.InstantCamera(tlf.CreateFirstDevice())

actual_num_of_foto = 0
print("Obsługa menu: \nq+ent -> Zakończ działanie programu\nt+ent ->Zrób zdjęcie i dodaj je do datasetu\n")
list_of_imgs = []
list_of_ratings = []
while True:
    war = input("Napisz co chcesz zrobić: ")
    if war == 't':
        cam.Open()
        cam.ExposureTimeAbs.SetValue(100000)
        cam.StartGrabbing()
        with cam.RetrieveResult(2000) as result:
            img.AttachGrabResultBuffer(result)
            filename = "%d.png" % actual_num_of_foto
            img.Save(pylon.ImageFileFormat_Png, filename)
            img.Release()
            actual_num_of_foto += 1
            rate = int(input("Podaj ocenę próbki:"))
            list_of_imgs.append(filename)
            list_of_ratings.append(rate)
        cam.StopGrabbing()
        cam.Close()
    if war == 'q':
        break
print(list_of_imgs)
print(list_of_ratings)

df = pd.DataFrame(
    {
        'img': list_of_imgs,
        'rating': list_of_ratings,
    }
)
df.to_csv('annotations.csv')
