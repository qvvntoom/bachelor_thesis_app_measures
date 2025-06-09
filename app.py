###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Preambuła
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

import streamlit as st
import pandas as pd
import numpy as np
import os
import time

import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer

from pingouin import multivariate_normality
from scipy.stats import chi2

import PyPDF2
import re

import cv2
from skimage.feature import hog
from PIL import Image

from io import BytesIO
from joblib import Parallel, delayed




###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Funkcje konwertujące i przygotowujące
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

def Przygotuj_tekst_do_analizy(Tekst):
    Tekst = Tekst.lower()
    Tekst = re.sub(r'[^\w\s]', '', Tekst)
    return Tekst

def Uzyskaj_tekst_z_PDF(file):
    Czytanie = PyPDF2.PdfReader(file)
    Tekst = ''
    for strona in range(len(Czytanie.pages)):
        Tekst += Czytanie.pages[strona].extract_text()
    return Tekst

def Histogram_kolorow_RGB_dla_obrazu(Obraz, Biny = 256):
    if len(Obraz.shape) == 2:
        Histogram = cv2.calcHist([Obraz], [0], None, [Biny], [0, 256]).flatten()
        Histogram /= np.sum(Histogram)
        return Histogram
    elif len(Obraz.shape) == 3:
        Czerwony, Zielony, Niebieski = cv2.split(Obraz)
        Histogram_czerwony = cv2.calcHist([Czerwony], [0], None, [Biny], [0, 256]).flatten()
        Histogram_zielony = cv2.calcHist([Zielony], [0], None, [Biny], [0, 256]).flatten()
        Histogram_niebieski = cv2.calcHist([Niebieski], [0], None, [Biny], [0, 256]).flatten()
        Histogram_kolorow = np.concatenate([Histogram_czerwony, Histogram_zielony, Histogram_niebieski])
        Histogram_kolorow /= np.sum(Histogram_kolorow)
        return Histogram_kolorow

def Histogram_HOG_dla_obrazu(Obraz):

    if len(Obraz.shape) == 3:
        Obraz_szary = cv2.cvtColor(Obraz, cv2.COLOR_RGB2GRAY)
    elif len(Obraz.shape) == 2:
        Obraz_szary = Obraz
    
    Cechy_HOG, _ = hog(Obraz_szary, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, feature_vector=True)
    
    return Cechy_HOG

def Notacja_szachowa_na_wspolrzedne(Notacja):
    Kolumna = ord(Notacja[0].upper()) - ord('A')
    Wiersz = Rozmiar_szachownicy - int(Notacja[1])
    return np.array([Wiersz, Kolumna])

def Kolorowanie_szachownicy(Wartosc_komorki, Indeks_wiersz, Indeks_kolumna):
    if (Indeks_wiersz + Indeks_kolumna) % 2 == 0:
        Kolor = "background-color: white; color: black;"
    else:
        Kolor = "background-color: black; color: white;"

    if Wartosc_komorki == "K":
        Kolor += " font-weight: bold; color: red;"
    elif Wartosc_komorki == "C":
        Kolor += " font-weight: bold; color: red;"

    return Kolor

def Styl_szachownicy(Dane):
    Styl = pd.DataFrame("", index=Dane.index, columns=Dane.columns)
    for Indeks_wiersz, Wiersz in enumerate(Dane.itertuples(index=False)):
        for Indeks_kolumna, Wartosc_komorki in enumerate(Wiersz):
            Styl.iloc[Indeks_wiersz, Indeks_kolumna] = Kolorowanie_szachownicy(Wartosc_komorki, Indeks_wiersz, Indeks_kolumna)
    return Styl


def Przetwarzaj_histogramy(Zaczytane_pliki):
    return Parallel(n_jobs=-1)(delayed(lambda Plik: (
        Histogram_kolorow_RGB_dla_obrazu(np.array(Image.open(Plik))),
        Histogram_HOG_dla_obrazu(np.array(Image.open(Plik)))
    ))(Plik) for Plik in Zaczytane_pliki)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Funkcje miar
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

def Odleglosc_Minkowskiego(X,Y,Parametr):
    return np.sum(np.abs(X - Y) ** Parametr) ** (1 / Parametr)

def Odleglosc_Manhattan(X, Y):
    return np.sum(np.abs(X - Y))


def Odleglosc_Euklidesa(X, Y):
    return np.sqrt(np.sum(np.abs(X - Y) ** 2))

def Kwadratowe_niepodobienstwo_Euklidesa(X,Y):
    return np.sum(np.abs(X - Y) ** 2)

def Odleglosc_Czebyszewa(X, Y):
    return np.max(np.abs(X - Y))

def Odleglosc_Canberry(X, Y):
    return np.sum(np.abs(X - Y)/(np.abs(X) + np.abs(Y)+ 1e-8))

def Niepodobienstwo_cosinusowe(X, Y):
    return 1 - np.dot(X, Y) / (np.linalg.norm(X) * np.linalg.norm(Y))

def Wazona_odleglosc_Euklidesa(X,Y,Wagi):
    return np.sqrt(np.sum(Wagi * (X - Y) ** 2))

def Odleglosc_Mahalanobisa(X, Y, Odwrotnosc_macierzy_kowariancji):
    return np.sqrt(np.dot(np.dot((X-Y), Odwrotnosc_macierzy_kowariancji), (X-Y).T))

def Odleglosc_Hamminga(X, Y):
    return np.sum(X != Y)

def Podobienstwo_Jaccarda(X, Y):
    intersection = np.sum(np.logical_and(X, Y))
    union = np.sum(np.logical_or(X, Y))
    return intersection / union if union != 0 else 0

def Niepodobienstwo_Jaccarda(X,Y):
    return 1 - Podobienstwo_Jaccarda(X,Y)

def Podobienstwo_Dicea(X, Y):
    intersection = np.sum(np.logical_and(X, Y))
    return (2 * intersection) / (np.sum(X) + np.sum(Y)) if (np.sum(X) + np.sum(Y)) != 0 else 0

def Niepodobienstwo_Dicea(X,Y):
    return 1 - Podobienstwo_Dicea(X,Y)

def Norma_Frobeniusa(Macierz):
    return np.sqrt(np.sum(Macierz**2))

def Niepodobienstwo_Prokrustesa(Zbior1, Zbior2):
    if Zbior1.shape != Zbior2.shape:
        raise ValueError("Zbiory nie mają takich samych wymiarów.")

    Srednia_Zbior1 = np.mean(Zbior1, axis=0)
    Srednia_Zbior2 = np.mean(Zbior2, axis=0)
    Wektor_odchylen_standardowych_Zbior1 = np.std(Zbior1, axis=0, ddof=1)
    Wektor_odchylen_standardowych_Zbior2 = np.std(Zbior2, axis=0, ddof=1)

    W1 = np.diag(Wektor_odchylen_standardowych_Zbior1)
    W2 = np.diag(Wektor_odchylen_standardowych_Zbior2)

    Zbior1_ST1 = (Zbior1 - Srednia_Zbior1) @ np.linalg.inv(W1)
    Zbior2_ST2 = (Zbior1 - Srednia_Zbior2) @ np.linalg.inv(W2)

    M_F = Zbior1_ST1.T @ Zbior2_ST2

    U, _, Vt = np.linalg.svd(M_F)
    Macierz_Rotacji = U @ Vt

    Argument_normy_Frobeniusa = Zbior1_ST1 @ Macierz_Rotacji - Zbior2_ST2
    Niepodobienstwo = Norma_Frobeniusa(Argument_normy_Frobeniusa)

    return Niepodobienstwo

def Niepodobienstwo_chi_kwadrat(X, Y):
    X = X / np.sum(X)
    Y = Y / np.sum(Y)
    return np.sum(((X - Y) ** 2) / (X + Y + 1e-8))

def Odleglosc_Hellingera(X, Y):
    X = X / np.sum(X)
    Y = Y / np.sum(Y)
    return (1 / np.sqrt(2)) * np.sqrt(np.sum((np.sqrt(X) - np.sqrt(Y)) ** 2))

def Odleglosc_Jensena_Shannona(X, Y):
    X = X / np.sum(X)
    Y = Y / np.sum(Y)
    Sredni_rozklad = (X + Y)/2
    return np.sqrt(0.5 * np.sum(X * np.log(X / Sredni_rozklad + 1e-8)) + 0.5 * np.sum(Y * np.log(Y / Sredni_rozklad + 1e-8)))

def Odleglosc_Jensena_Shannona(X, Y):
    X = X / np.sum(X)
    Y = Y / np.sum(Y)

    Sredni_rozklad = (X + Y) / 2

    Maska_X = X > 0
    Maska_Y = Y > 0
    Maska_Sredni_rozklad = Sredni_rozklad > 0

    log_X = np.zeros_like(X)
    log_Y = np.zeros_like(Y)

    log_X[Maska_X & Maska_Sredni_rozklad] = np.log2(X[Maska_X & Maska_Sredni_rozklad] / Sredni_rozklad[Maska_X & Maska_Sredni_rozklad])
    log_Y[Maska_Y & Maska_Sredni_rozklad] = np.log2(Y[Maska_Y & Maska_Sredni_rozklad] / Sredni_rozklad[Maska_Y & Maska_Sredni_rozklad])

    return np.sqrt(0.5 * np.sum(X * log_X) + 0.5 * np.sum(Y * log_Y))

def Zmodyfikowane_niepodobienstwo_chi_kwadrat(X, Y, H):
    X_przetworzone = np.sqrt(X / (X + H + 1e-8))
    Y_przetworzone = np.sqrt(Y / (Y + H + 1e-8))
    return Kwadratowe_niepodobienstwo_Euklidesa(X_przetworzone,Y_przetworzone)

def Odleglosc_Levenshteina(Slowo_1, Slowo_2):
    m, n = len(Slowo_1), len(Slowo_2)
    Macierz_programowania_dynamicznego = np.zeros((m + 1, n + 1), dtype=int)

    for i in range(m + 1):
        Macierz_programowania_dynamicznego[i][0] = i
    for j in range(n + 1):
        Macierz_programowania_dynamicznego[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            Koszt = 0 if Slowo_1[i - 1] == Slowo_2[j - 1] else 1
            Macierz_programowania_dynamicznego[i][j] = min(
                Macierz_programowania_dynamicznego[i - 1][j] + 1,
                Macierz_programowania_dynamicznego[i][j - 1] + 1,
                Macierz_programowania_dynamicznego[i - 1][j - 1] + Koszt
            )

    Wiersze = [''] + [Slowo_1[:i] for i in range(1, m + 1)]
    Kolumny = [''] + [Slowo_2[:j] for j in range(1, n + 1)]

    Macierz_programowania_dynamicznego_z_podciagami = pd.DataFrame(Macierz_programowania_dynamicznego, index=Wiersze, columns=Kolumny)
    return Macierz_programowania_dynamicznego_z_podciagami, Macierz_programowania_dynamicznego[m][n]

def Odleglosc_Mahalanobisa(X, Wektor_srednich, Odwrotnosc_macierzy_kowariancji):
    delta = X - Wektor_srednich
    return np.sqrt(np.dot(np.dot(delta, Odwrotnosc_macierzy_kowariancji), delta.T))

def Narysuj_punkty_i_granice_decyzyjne_odleglosci_Mahalanobisa(Dane, Wektor_srednich, Macierz_kowariancji, Zadane_prawdopodobienstwo_1_alpha):
    Wymiary = Dane.shape[1]
    Wartosci_wlasne, Wektory_wlasne = np.linalg.eig(Macierz_kowariancji)
    Lambdy = np.sqrt(Wartosci_wlasne)

    if Wymiary == 2:
        Katy = np.linspace(0, 2 * np.pi, 100)
        Sfera = np.array([np.cos(Katy), np.sin(Katy)])

        Elipsoida = np.sqrt(chi2.ppf(Zadane_prawdopodobienstwo_1_alpha, Wymiary))*np.diag(Lambdy)@Sfera
        Elipsoida_obrocona_i_przesunieta = Wektory_wlasne@Elipsoida + Wektor_srednich[:, np.newaxis]

    if Wymiary == 3:
        U = np.linspace(0, 2 * np.pi, 100)
        V = np.linspace(0, np.pi, 100)

        Sfera = np.array([np.outer(np.cos(U), np.sin(V)).flatten(),
                          np.outer(np.sin(U), np.sin(V)).flatten(),
                          np.outer(np.ones_like(U), np.cos(V)).flatten()])

        Elipsoida = np.sqrt(chi2.ppf(Zadane_prawdopodobienstwo_1_alpha, Wymiary)) * np.diag(Lambdy)@Sfera
        Elipsoida_obrocona_i_przesunieta = Wektory_wlasne@Elipsoida + Wektor_srednich[:, np.newaxis]

        X, Y, Z = Elipsoida_obrocona_i_przesunieta[0].reshape(100, 100), Elipsoida_obrocona_i_przesunieta[1].reshape(100, 100), Elipsoida_obrocona_i_przesunieta[2].reshape(100, 100)

        Elipsoida_obrocona_i_przesunieta = (X,Y,Z)

    return Wartosci_wlasne, Wektory_wlasne, Lambdy, Elipsoida_obrocona_i_przesunieta


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Funkcje algorytmów
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

def Algorytm_K_Srednich(Zbior_danych, Etykiety, Proporcja_testowy, Miara_niepodobieństwa, Maksymalna_liczba_iteracji, Tolerancja, Liczba_losowa = 42):

    Ilosc_klastrow = len(np.unique(Etykiety))
    Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe = train_test_split(Zbior_danych,
                                                                                                                Etykiety,
                                                                                                                test_size = Proporcja_testowy,
                                                                                                                random_state = Liczba_losowa)

    Czas_start = time.time()
    np.random.seed(Liczba_losowa)
    Centroidy = Zbior_danych_treningowych[np.random.choice(Zbior_danych_treningowych.shape[0], size=Ilosc_klastrow, replace=False)]

    for Iteracja in range(Maksymalna_liczba_iteracji):
        Etykiety_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid) for centroid in Centroidy]) for X in Zbior_danych_treningowych])

        Nowe_centroidy = np.array([Zbior_danych_treningowych[Etykiety_algorytm == k].mean(axis=0) for k in range(Ilosc_klastrow)])

        if np.all(np.abs(Nowe_centroidy - Centroidy) < Tolerancja):
            break

        Centroidy = Nowe_centroidy

    Mapowanie_etykiet = {}
    for klaster in range(Ilosc_klastrow):
        Prawdziwe_etykiety = np.ravel(Etykiety_treningowe[Etykiety_algorytm == klaster])
        if len(Prawdziwe_etykiety) > 0:
            Prawdziwe_etykiety = Prawdziwe_etykiety.astype(int)
            Mapowanie_etykiet[klaster] = np.bincount(Prawdziwe_etykiety).argmax()
        else:
            Mapowanie_etykiet[klaster] = -1

    Etykiety_algorytm_zmapowane = np.array([Mapowanie_etykiet[etykieta] for etykieta in Etykiety_algorytm])

    Dokladnosc_treningowa = accuracy_score(Etykiety_treningowe, Etykiety_algorytm_zmapowane)

    Etykiety_testowe_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid) for centroid in Centroidy]) for X in Zbior_danych_testowych])
    Etykiety_testowe_algorytm_zmapowane = np.array([Mapowanie_etykiet.get(etykieta, -1) for etykieta in Etykiety_testowe_algorytm])

    Dokladnosc_testowa = accuracy_score(Etykiety_testowe, Etykiety_testowe_algorytm_zmapowane)

    Czas_stop = time.time()
    Czas_wykonywania = Czas_stop - Czas_start

    return Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania

def Algorytm_K_Srednich_Minkowski(Zbior_danych, Etykiety, Parametr, Proporcja_testowy, Maksymalna_liczba_iteracji, Tolerancja, Liczba_losowa = 42):

    Ilosc_klastrow = len(np.unique(Etykiety))
    Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe = train_test_split(Zbior_danych,
                                                                                                                Etykiety,
                                                                                                                test_size = Proporcja_testowy,
                                                                                                                random_state = Liczba_losowa)

    Czas_start = time.time()
    np.random.seed(Liczba_losowa)
    Centroidy = Zbior_danych_treningowych[np.random.choice(Zbior_danych_treningowych.shape[0], size=Ilosc_klastrow, replace=False)]

    for Iteracja in range(Maksymalna_liczba_iteracji):
        Etykiety_algorytm = np.array([np.argmin([Odleglosc_Minkowskiego(X, centroid, Parametr) for centroid in Centroidy]) for X in Zbior_danych_treningowych])

        Nowe_centroidy = np.array([Zbior_danych_treningowych[Etykiety_algorytm == k].mean(axis=0) for k in range(Ilosc_klastrow)])

        if np.all(np.abs(Nowe_centroidy - Centroidy) < Tolerancja):
            break

        Centroidy = Nowe_centroidy

    Mapowanie_etykiet = {}
    for klaster in range(Ilosc_klastrow):
        Prawdziwe_etykiety = np.ravel(Etykiety_treningowe[Etykiety_algorytm == klaster])
        if len(Prawdziwe_etykiety) > 0:
            Prawdziwe_etykiety = Prawdziwe_etykiety.astype(int)
            Mapowanie_etykiet[klaster] = np.bincount(Prawdziwe_etykiety).argmax()
        else:
            Mapowanie_etykiet[klaster] = -1

    Etykiety_algorytm_zmapowane = np.array([Mapowanie_etykiet[etykieta] for etykieta in Etykiety_algorytm])

    Dokladnosc_treningowa = accuracy_score(Etykiety_treningowe, Etykiety_algorytm_zmapowane)

    Etykiety_testowe_algorytm = np.array([np.argmin([Odleglosc_Minkowskiego(X, centroid, Parametr) for centroid in Centroidy]) for X in Zbior_danych_testowych])
    Etykiety_testowe_algorytm_zmapowane = np.array([Mapowanie_etykiet.get(etykieta, -1) for etykieta in Etykiety_testowe_algorytm])

    Dokladnosc_testowa = accuracy_score(Etykiety_testowe, Etykiety_testowe_algorytm_zmapowane)

    Czas_stop = time.time()
    Czas_wykonywania = Czas_stop - Czas_start

    return Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania

def Algorytm_K_Srednich_Znormalizowana(Zbior_danych, Etykiety, Proporcja_testowy, Miara_niepodobieństwa, Wagi, Maksymalna_liczba_iteracji, Tolerancja, Liczba_losowa = 42):

    Ilosc_klastrow = len(np.unique(Etykiety))
    Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe = train_test_split(Zbior_danych,
                                                                                                                Etykiety,
                                                                                                                test_size = Proporcja_testowy,
                                                                                                                random_state = Liczba_losowa)

    Czas_start = time.time()
    np.random.seed(Liczba_losowa)
    Centroidy = Zbior_danych_treningowych[np.random.choice(Zbior_danych_treningowych.shape[0],
                                                           size=Ilosc_klastrow,
                                                           replace=False)]

    for Iteracja in range(Maksymalna_liczba_iteracji):
        Etykiety_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid, Wagi) for centroid in Centroidy]) for X in Zbior_danych_treningowych])

        Nowe_centroidy = np.array([Zbior_danych_treningowych[Etykiety_algorytm == k].mean(axis=0) for k in range(Ilosc_klastrow)])

        if np.all(np.abs(Nowe_centroidy - Centroidy) < Tolerancja):
            break

        Centroidy = Nowe_centroidy

    Mapowanie_etykiet = {}
    for klaster in range(Ilosc_klastrow):
        Prawdziwe_etykiety = np.ravel(Etykiety_treningowe[Etykiety_algorytm == klaster])
        if len(Prawdziwe_etykiety) > 0:
            Prawdziwe_etykiety = Prawdziwe_etykiety.astype(int)
            Mapowanie_etykiet[klaster] = np.bincount(Prawdziwe_etykiety).argmax()
        else:
            Mapowanie_etykiet[klaster] = -1

    Etykiety_algorytm_zmapowane = np.array([Mapowanie_etykiet[etykieta] for etykieta in Etykiety_algorytm])

    Dokladnosc_treningowa = accuracy_score(Etykiety_treningowe, Etykiety_algorytm_zmapowane)

    Etykiety_testowe_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid, Wagi) for centroid in Centroidy]) for X in Zbior_danych_testowych])
    Etykiety_testowe_algorytm_zmapowane = np.array([Mapowanie_etykiet.get(etykieta, -1) for etykieta in Etykiety_testowe_algorytm])

    Dokladnosc_testowa = accuracy_score(Etykiety_testowe, Etykiety_testowe_algorytm_zmapowane)

    Czas_stop = time.time()
    Czas_wykonywania = Czas_stop - Czas_start

    return Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania


def Wizualizacja_klastrow(Dane, Etykiety, Centroidy, Ilosc_klastrow, Miara_niepodobienstwa, Paleta_kolorow='Viridis'):
    Wymiary = Dane.shape[1]
    Etykiety = Etykiety.ravel()
    if Wymiary == 2:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),np.linspace(y_min, y_max, 500))
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel()]

        Etykiety_siatki = np.array([np.argmin([Miara_niepodobienstwa(Punkt, Centroid) for Centroid in Centroidy])for Punkt in Siatka_punktow])

        Fig = go.Figure()

        Fig.add_trace(go.Contour(x = np.linspace(x_min, x_max, 500), y = np.linspace(y_min, y_max, 500), z = Etykiety_siatki.reshape(xx.shape),
            colorscale=Paleta_kolorow,
            showscale=False,
            opacity=0.3))

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter(x = Dane[Maska, 0], y = Dane[Maska, 1],
                mode='markers',
                marker=dict(size=7, color=Klaster, colorscale=Paleta_kolorow, opacity=0.8),
                name=f"Klaster {Klaster}"
            ))

        Fig.add_trace(go.Scatter(x=Centroidy[:, 0], y=Centroidy[:, 1],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig

    elif Wymiary == 3:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
        z_min, z_max = Dane[:, 2].min() - 1, Dane[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30), np.linspace(z_min, z_max, 30),)
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        Etykiety_siatki = np.array([np.argmin([Miara_niepodobienstwa(Punkt, Centroid) for Centroid in Centroidy]) for Punkt in Siatka_punktow])

        Fig = go.Figure()

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter3d(x = Dane[Maska, 0], y = Dane[Maska, 1], z = Dane[Maska, 2],
                mode='markers',
                marker=dict(size=5, color=Klaster, colorscale=Paleta_kolorow, opacity=1),
                name=f"Klaster {Klaster}"))

        Fig.add_trace(go.Scatter3d(x = Centroidy[:, 0], y = Centroidy[:, 1], z = Centroidy[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.add_trace(go.Scatter3d(x = Siatka_punktow[:, 0], y = Siatka_punktow[:, 1], z = Siatka_punktow[:, 2],
            mode='markers',
            marker=dict(size=1.8, color=Etykiety_siatki, colorscale=Paleta_kolorow, opacity=0.1),
            name="Obszary decyzyjne"))

        Fig.update_layout(scene=dict(), legend=dict(bgcolor= "rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig

def Wizualizacja_klastrow_Minkowski(Dane, Etykiety, Centroidy, Ilosc_klastrow, Parametr, Paleta_kolorow='Viridis'):
    Wymiary = Dane.shape[1]
    Etykiety = Etykiety.ravel()
    if Wymiary == 2:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel()]

        Etykiety_siatki = np.array([np.argmin([Odleglosc_Minkowskiego(Punkt, Centroid, Parametr) for Centroid in Centroidy]) for Punkt in Siatka_punktow])

        Fig = go.Figure()

        Fig.add_trace(go.Contour(x = np.linspace(x_min, x_max, 500), y = np.linspace(y_min, y_max, 500), z = Etykiety_siatki.reshape(xx.shape),
            colorscale=Paleta_kolorow,
            showscale=False,
            opacity=0.3))

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter(x = Dane[Maska, 0], y = Dane[Maska, 1],
                mode='markers',
                marker=dict(size=7, color=Klaster, colorscale=Paleta_kolorow, opacity=0.8),
                name=f"Klaster {Klaster}"))

        Fig.add_trace(go.Scatter(x = Centroidy[:, 0], y = Centroidy[:, 1],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig

    elif Wymiary == 3:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
        z_min, z_max = Dane[:, 2].min() - 1, Dane[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30), np.linspace(z_min, z_max, 30),)
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        Etykiety_siatki = np.array([np.argmin([Odleglosc_Minkowskiego(Punkt, Centroid, Parametr) for Centroid in Centroidy]) for Punkt in Siatka_punktow])

        Fig = go.Figure()

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter3d(x = Dane[Maska, 0], y = Dane[Maska, 1], z = Dane[Maska, 2],
                mode='markers',
                marker=dict(size=5, color=Klaster, colorscale=Paleta_kolorow, opacity=1),
                name=f"Klaster {Klaster}"))

        Fig.add_trace(go.Scatter3d(x = Centroidy[:, 0], y = Centroidy[:, 1], z = Centroidy[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.add_trace(go.Scatter3d(x = Siatka_punktow[:, 0], y = Siatka_punktow[:, 1], z = Siatka_punktow[:, 2],
            mode='markers',
            marker=dict(size=1.8, color=Etykiety_siatki, colorscale=Paleta_kolorow, opacity=0.1),
            name="Obszary decyzyjne"))

        Fig.update_layout(scene=dict(), legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig


def Wizualizacja_klastrow_Znormalizowana(Dane, Etykiety, Centroidy, Ilosc_klastrow, Wagi, Paleta_kolorow='Viridis'):
    Wymiary = Dane.shape[1]
    Etykiety = Etykiety.ravel()
    if Wymiary == 2:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1

        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel()]

        Etykiety_siatki = np.array([np.argmin([Wazona_odleglosc_Euklidesa(Punkt, Centroid, Wagi) for Centroid in Centroidy]) for Punkt in Siatka_punktow])

        Fig = go.Figure()

        Fig.add_trace(go.Contour(x = np.linspace(x_min, x_max, 500), y = np.linspace(y_min, y_max, 500), z = Etykiety_siatki.reshape(xx.shape),
            colorscale=Paleta_kolorow,
            showscale=False,
            opacity=0.3))

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter(x = Dane[Maska, 0], y = Dane[Maska, 1],
                mode='markers',
                marker=dict(size=7, color=Klaster, colorscale=Paleta_kolorow, opacity=0.8),
                name=f"Klaster {Klaster}"))

        Fig.add_trace(go.Scatter(x = Centroidy[:, 0], y = Centroidy[:, 1],
            mode='markers',
            marker=dict(size=10, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig

    elif Wymiary == 3:
        x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
        y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
        z_min, z_max = Dane[:, 2].min() - 1, Dane[:, 2].max() + 1

        xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30), np.linspace(z_min, z_max, 30),)
        Siatka_punktow = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]

        Etykiety_siatki = np.array([np.argmin([Wazona_odleglosc_Euklidesa(Punkt, Centroid, Wagi) for Centroid in Centroidy]) for Punkt in Siatka_punktow])

        Fig = go.Figure()

        for Klaster in range(Ilosc_klastrow):
            Maska = Etykiety == Klaster
            Fig.add_trace(go.Scatter3d(x = Dane[Maska, 0], y = Dane[Maska, 1], z = Dane[Maska, 2],
                mode='markers',
                marker=dict(size=5, color=Klaster, colorscale=Paleta_kolorow, opacity=1.0),
                name=f"Klaster {Klaster}"))

        Fig.add_trace(go.Scatter3d(x = Centroidy[:, 0], y = Centroidy[:, 1], z = Centroidy[:, 2],
            mode='markers',
            marker=dict(size=4, color='red', symbol='x', opacity=1.0),
            name="Centroidy"))

        Fig.add_trace(go.Scatter3d(x = Siatka_punktow[:, 0], y = Siatka_punktow[:, 1], z = Siatka_punktow[:, 2],
            mode='markers',
            marker=dict(size=1.8, color=Etykiety_siatki, colorscale=Paleta_kolorow, opacity=0.1),
            name="Obszary decyzyjne"))

        Fig.update_layout(scene=dict(), legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)

        return Fig


###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Początek apki
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

st.set_page_config(
    page_title="Miary podobieństwa",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Twoje obecne style
st.markdown(
    """
    <style>
    :root {
        --primary-color: #001BFF;
        --background-color: #FFFFFF;
        --secondary-background-color: #EAEAEA;
        --text-color: #000000;
    }
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    .stButton button {
        background-color: var(--primary-color);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
# Dodatkowy styl do ukrycia paska
st.markdown(
    """
    <style>
    div.block-container {
        padding-top: 0 !important;
    }
    header, .fullscreen-header {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Praktyczne zastosowanie miar podobieństwa i niepodobieństwa")

st.sidebar.title("🧭 Nawigacja")
Opcja_pracy_w_aplikacji = st.sidebar.selectbox("Wybierz",
    ["Witaj w aplikacji!",
    "Klasteryzacja i klasyfikacja danych ogólnych: dane i etykiety w formacie CSV/XLS/XLSX",
    "Analiza podobieństwa obrazów: dane formacie JPG/JPEG/PNG",
    "Analiza podobieństwa dokumentów: dane w formacie PDF",
    "Wykrywanie wartości odstających w zbiorze danych: dane w formacie CSV/XLS/XLSX",
    "Analiza podobieństwa danych binarnych: dane w formacie CSV/XLS/XLSX",
    "Analiza niepodobieństwa Prokrustesa zbiorów punktów: dane w formacie CSV/XLS/XLSX",
    "Wyliczanie odległości Levenshteina pomiędzy ciągami tekstowymi",
    "Ilość ruchów królem na szachownicy",
    "Kod algorytmów i funkcji"])


if Opcja_pracy_w_aplikacji == "Witaj w aplikacji!":

    st.markdown("""
    Niniejsza aplikacja stanowi część praktyczną pracy inżynierskiej pt. **„Miary podobieństwa i niepodobieństwa w analizie danych: podstawy matematyczne, zastosowania, implementacja”**, przygotowanej na kierunku *Matematyka stosowana* (specjalność: *Analityka danych*) na Wydziale Informatyki i Telekomunikacji Politechniki Krakowskiej.

    Celem aplikacji jest umożliwienie interaktywnego zapoznania się z wybranymi miarami podobieństwa i niepodobieństwa wykorzystywanymi w analizie danych. Prezentowane moduły pozwalają przeprowadzać eksperymenty z różnymi typami danych – od wektorów liczbowych, przez obrazy, aż po teksty i ciągi znaków.

    Zaimplementowane funkcjonalności obejmują m.in.:
    - porównywanie obserwacji przy użyciu klasycznych i specjalistycznych metryk,
    - analizę klasteryzacji i klasyfikacji na podstawie wybranych miar,
    - ocenę podobieństwa obrazów i dokumentów tekstowych,
    - wykrywanie obserwacji odstających,
    - analizę danych binarnych i sekwencyjnych.

    Aplikacja powstała z myślą o użytkownikach zainteresowanych eksploracją danych, analizą statystyczną oraz uczeniem maszynowym. Każdy moduł opatrzony jest komentarzami i przykładami, co czyni narzędzie przystępnym zarówno dla studentów, jak i osób pracujących w obszarze analityki danych.

    Autor: **Patryk Doniec**  
    Promotorzy: **dr Marcin Skrzyński, mgr Szymon Sroka**  
    Politechnika Krakowska, 2025
    """)


#with Kolumna_1:
    #st.subheader("Wybierz jedną z dostępnych pozycji i dowiedz się więcej:")
    #st.write("")
    #with st.expander("Opcje dostępne na liście"):
        #st.text("- Klasteryzacja i klasyfikacja danych ogólnych: dane i etykiety w formacie CSV/XLS/XLSX")
        #st.text("- Analiza podobieństwa obrazów: dane formacie JPG/JPEG/PNG")
        #st.text("- Analiza podobieństwa dokumentów: dane w formacie PDF")
        #st.text("- Wykrywanie wartości odstających w zbiorze danych: dane w formacie CSV/XLS/XLSX")
        #st.text("- Analiza podobieństwa danych binarnych: dane w formacie CSV/XLS/XLSX")
        #st.text("- Analiza niepodobieństwa Prokrustesa zbiorów punktów: dane w formacie CSV/XLS/XLSX")
        #st.text("- Wyliczanie odległości Levenshteina pomiędzy ciągami tekstowymi")
        #st.text("- Ilość ruchów królem na szachownicy")

#with Kolumna_2:



if Opcja_pracy_w_aplikacji == "Kod algorytmów i funkcji":
    with st.expander("Pakiety w aplikacji"):
        st.markdown("""
        ```python
            import streamlit as st
        import pandas as pd
        import numpy as np
        import os
        import time
        import plotly.graph_objects as go
        import PyPDF2
        import re
        import cv2

        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        from sklearn.feature_extraction.text import CountVectorizer
        from pingouin import multivariate_normality
        from scipy.stats import chi2
        from skimage.feature import hog
        from PIL import Image
        from io import BytesIO
        from joblib import Parallel, delayed
""")
    with st.expander("Algorytm K-Średnich"):
        st.markdown("""
        ```python
                    def Algorytm_K_Srednich(Zbior_danych, Etykiety, Proporcja_testowy, Miara_niepodobieństwa, Maksymalna_liczba_iteracji, Tolerancja, Liczba_losowa):
                        Ilosc_klastrow = len(np.unique(Etykiety))
                        Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe = train_test_split(Zbior_danych,
                                                                                                                                    Etykiety,
                                                                                                                                    test_size = Proporcja_testowy,
                                                                                                                                    random_state = Liczba_losowa)
                        Czas_start = time.time()
                        np.random.seed(Liczba_losowa)
                        Centroidy = Zbior_danych_treningowych[np.random.choice(Zbior_danych_treningowych.shape[0], size=Ilosc_klastrow, replace=False)]
                        for Iteracja in range(Maksymalna_liczba_iteracji):
                            Etykiety_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid) for centroid in Centroidy]) for X in Zbior_danych_treningowych])
                            Nowe_centroidy = np.array([Zbior_danych_treningowych[Etykiety_algorytm == k].mean(axis=0) for k in range(Ilosc_klastrow)])
                            if np.all(np.abs(Nowe_centroidy - Centroidy) < Tolerancja):
                                break
                            Centroidy = Nowe_centroidy
                        Mapowanie_etykiet = {}
                        for klaster in range(Ilosc_klastrow):
                            Prawdziwe_etykiety = np.ravel(Etykiety_treningowe[Etykiety_algorytm == klaster])
                            if len(Prawdziwe_etykiety) > 0:
                                Prawdziwe_etykiety = Prawdziwe_etykiety.astype(int)
                                Mapowanie_etykiet[klaster] = np.bincount(Prawdziwe_etykiety).argmax()
                            else:
                                Mapowanie_etykiet[klaster] = -1
                        Etykiety_algorytm_zmapowane = np.array([Mapowanie_etykiet[etykieta] for etykieta in Etykiety_algorytm])
                        Dokladnosc_treningowa = accuracy_score(Etykiety_treningowe, Etykiety_algorytm_zmapowane)
                        Etykiety_testowe_algorytm = np.array([np.argmin([Miara_niepodobieństwa(X, centroid) for centroid in Centroidy]) for X in Zbior_danych_testowych])
                        Etykiety_testowe_algorytm_zmapowane = np.array([Mapowanie_etykiet.get(etykieta, -1) for etykieta in Etykiety_testowe_algorytm])
                        Dokladnosc_testowa = accuracy_score(Etykiety_testowe, Etykiety_testowe_algorytm_zmapowane)
                        Czas_stop = time.time()
                        Czas = Czas_stop - Czas_start
                        return Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas
""")

    with st.expander("Wizualizacja klastów"):
        st.markdown("""
        ```python
                    def Wizualizacja_klastrow(Dane, Etykiety, Centroidy, Ilosc_klastrow, Miara_niepodobienstwa, Paleta_kolorow='Viridis'):
                        Wymiary = Dane.shape[1]
                        Etykiety = Etykiety.ravel()
                        if Wymiary == 2:
                            x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
                            y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
                            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500),np.linspace(y_min, y_max, 500))
                            Siatka_punktow = np.c_[xx.ravel(), yy.ravel()]
                            Etykiety_siatki = np.array([np.argmin([Miara_niepodobienstwa(Punkt, Centroid) for Centroid in Centroidy])for Punkt in Siatka_punktow])
                            Fig = go.Figure()
                            Fig.add_trace(go.Contour(x = np.linspace(x_min, x_max, 500), y = np.linspace(y_min, y_max, 500), z = Etykiety_siatki.reshape(xx.shape),
                                colorscale=Paleta_kolorow, showscale=False, opacity=0.3))
                            for Klaster in range(Ilosc_klastrow):
                                Maska = Etykiety == Klaster
                                Fig.add_trace(go.Scatter(x = Dane[Maska, 0], y = Dane[Maska, 1], mode='markers', 
                                    marker=dict(size=7, color=Klaster, colorscale=Paleta_kolorow, opacity=0.8), name=f"Klaster {Klaster}"))
                            Fig.add_trace(go.Scatter(x=Centroidy[:, 0], y=Centroidy[:, 1], mode='markers', marker=dict(size=10, color='red', symbol='x', opacity=1.0),
                                name="Centroidy"))
                            Fig.update_layout(legend=dict(bgcolor="rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)
                            return Fig
                        elif Wymiary == 3:
                            x_min, x_max = Dane[:, 0].min() - 1, Dane[:, 0].max() + 1
                            y_min, y_max = Dane[:, 1].min() - 1, Dane[:, 1].max() + 1
                            z_min, z_max = Dane[:, 2].min() - 1, Dane[:, 2].max() + 1
                            xx, yy, zz = np.meshgrid(np.linspace(x_min, x_max, 30), np.linspace(y_min, y_max, 30), np.linspace(z_min, z_max, 30),)
                            Siatka_punktow = np.c_[xx.ravel(), yy.ravel(), zz.ravel()]
                            Etykiety_siatki = np.array([np.argmin([Miara_niepodobienstwa(Punkt, Centroid) for Centroid in Centroidy]) for Punkt in Siatka_punktow])
                            Fig = go.Figure()
                            for Klaster in range(Ilosc_klastrow):
                                Maska = Etykiety == Klaster
                                Fig.add_trace(go.Scatter3d(x = Dane[Maska, 0], y = Dane[Maska, 1], z = Dane[Maska, 2], mode='markers',
                                    marker=dict(size=5, color=Klaster, colorscale=Paleta_kolorow, opacity=1), name=f"Klaster {Klaster}"))
                            Fig.add_trace(go.Scatter3d(x = Centroidy[:, 0], y = Centroidy[:, 1], z = Centroidy[:, 2], mode='markers',
                                marker=dict(size=4, color='red', symbol='x', opacity=1.0), name="Centroidy"))
                            Fig.add_trace(go.Scatter3d(x = Siatka_punktow[:, 0], y = Siatka_punktow[:, 1], z = Siatka_punktow[:, 2], mode='markers',
                                marker=dict(size=1.8, color=Etykiety_siatki, colorscale=Paleta_kolorow, opacity=0.1), name="Obszary decyzyjne"))
                            Fig.update_layout(scene=dict(), legend=dict(bgcolor= "rgba(255,255,255,0.5)", bordercolor="Black", borderwidth=1), width=1200, height=800)
                            return Fig
""")

    
    
    with st.expander("Funkcja dla histogramów RGB i HOG"):
        st.markdown("""
        ```python
            def Histogram_RGB(Obraz, Biny = 256):
            if len(Obraz.shape) == 2:
                Histogram = cv2.calcHist([Obraz], [0], None, [Biny], [0, 256]).flatten()
                Histogram /= np.sum(Histogram)
                return Histogram
            elif len(Obraz.shape) == 3:
                Czerwony, Zielony, Niebieski = cv2.split(Obraz)
                Histogram_czerwony = cv2.calcHist([Czerwony], [0], None, [Biny], [0, 256]).flatten()
                Histogram_zielony = cv2.calcHist([Zielony], [0], None, [Biny], [0, 256]).flatten()
                Histogram_niebieski = cv2.calcHist([Niebieski], [0], None, [Biny], [0, 256]).flatten()
                Histogram_kolorow = np.concatenate([Histogram_czerwony, Histogram_zielony, Histogram_niebieski])
                Histogram_kolorow /= np.sum(Histogram_kolorow)
                return Histogram_kolorow
        def Histogram_HOG(Obraz)
            if len(Obraz.shape) == 3:
                Obraz_szary = cv2.cvtColor(Obraz, cv2.COLOR_RGB2GRAY)
            elif len(Obraz.shape) == 2:
                Obraz_szary = Obraz
            Hist_HOG, _ = hog(Obraz_szary, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True, feature_vector=True)
            return Hist_HOG

                    
""")
    with st.expander("Uzyskaj tekst z PDF"):
        st.markdown("""
        ```python
                    def Uzyskaj_tekst_z_PDF(file):
                        Czytanie = PyPDF2.PdfReader(file)
                        Tekst = ''
                        for strona in range(len(Czytanie.pages)):
                            Tekst += Czytanie.pages[strona].extract_text()
                        return Tekst
""")
    
    with st.expander("Funkcja zwracająca dane do wykresu dla odległości Mahalanobisa"):
        st.markdown("""
        ```python
                    def Narysuj_punkty_i_granice_decyzyjne_odleglosci_Mahalanobisa(Dane, Wektor_srednich, Macierz_kowariancji, Zadane_prawdopodobienstwo_1_alpha):
                        Wymiary = Dane.shape[1]
                        Wartosci_wlasne, Wektory_wlasne = np.linalg.eig(Macierz_kowariancji)
                        Lambdy = np.sqrt(Wartosci_wlasne)
                        if Wymiary == 2:
                            Katy = np.linspace(0, 2 * np.pi, 100)
                            Sfera = np.array([np.cos(Katy), np.sin(Katy)])
                            Elipsoida = np.sqrt(chi2.ppf(Zadane_prawdopodobienstwo_1_alpha, Wymiary))*np.diag(Lambdy)@Sfera
                            Elipsoida_obrocona_i_przesunieta = Wektory_wlasne@Elipsoida + Wektor_srednich[:, np.newaxis]
                        if Wymiary == 3:
                            U = np.linspace(0, 2 * np.pi, 100)
                            V = np.linspace(0, np.pi, 100)
                            Sfera = np.array([np.outer(np.cos(U), np.sin(V)).flatten(),
                                            np.outer(np.sin(U), np.sin(V)).flatten(),
                                            np.outer(np.ones_like(U), np.cos(V)).flatten()])
                            Elipsoida = np.sqrt(chi2.ppf(Zadane_prawdopodobienstwo_1_alpha, Wymiary)) * np.diag(Lambdy)@Sfera
                            Elipsoida_obrocona_i_przesunieta = Wektory_wlasne@Elipsoida + Wektor_srednich[:, np.newaxis]
                            X, Y, Z = Elipsoida_obrocona_i_przesunieta[0].reshape(100, 100), Elipsoida_obrocona_i_przesunieta[1].reshape(100, 100), Elipsoida_obrocona_i_przesunieta[2].reshape(100, 100)
                            Elipsoida_obrocona_i_przesunieta = (X,Y,Z)
                        return Wartosci_wlasne, Wektory_wlasne, Lambdy, Elipsoida_obrocona_i_przesunieta
""")

    with st.expander("Odległość Levenshteina"):
        st.markdown("""
        ```python
                    def Odleglosc_Levenshteina(Slowo_1, Slowo_2):
                        m, n = len(Slowo_1), len(Slowo_2)
                        Macierz_programowania_dynamicznego = np.zeros((m + 1, n + 1), dtype=int)
                        for i in range(m + 1):
                            Macierz_programowania_dynamicznego[i][0] = i
                        for j in range(n + 1):
                            Macierz_programowania_dynamicznego[0][j] = j
                        for i in range(1, m + 1):
                            for j in range(1, n + 1):
                                Koszt = 0 if Slowo_1[i - 1] == Slowo_2[j - 1] else 1
                                Macierz_programowania_dynamicznego[i][j] = min(
                                    Macierz_programowania_dynamicznego[i - 1][j] + 1,
                                    Macierz_programowania_dynamicznego[i][j - 1] + 1,
                                    Macierz_programowania_dynamicznego[i - 1][j - 1] + Koszt
                                )
                        Wiersze = [''] + [Slowo_1[:i] for i in range(1, m + 1)]
                        Kolumny = [''] + [Slowo_2[:j] for j in range(1, n + 1)]
                        Macierz_programowania_dynamicznego_z_podciagami = pd.DataFrame(Macierz_programowania_dynamicznego, index=Wiersze, columns=Kolumny)
                        return Macierz_programowania_dynamicznego_z_podciagami, Macierz_programowania_dynamicznego[m][n]
""")


elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa obrazów: dane formacie JPG/JPEG/PNG":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia analizę podobieństwa obrazów w formacie JPG lub JEPG lub PNG.

        - W pierwszej kolejności zostaną załadowane pliki aplikacji, a następnie zostaną one przetworzone na histogramy RGB, oraz histogramy HOG.
        - Następnie użytkownik podejmuje wybór czy chce skorzystać z własnych plików, które należy wgrać do aplikacji, czy chce wykonać test na plikach aplikacji.
        - Pliki wgrane w aplikacje obejmują następujące zbiory obrazów:
            - Cyfry MNIST:
                - Zbiór zawierający obrazy cyfr napisanych odręcznie (0 do 9)
                - Obrazy w skali szarości o rozmiarze 28 na 28 pikseli

            - Znaki drogowe TS Belgium:
                - Zbiór danych zawierający obrazy znaków drogowych zebranych w Belgii
                - Kolorowe obrazy znaków drogowych o różnej rozdzielczości

        - Po wybraniu jedengo z plików aplikacji, lub załadowaniu własnych plików w formacie JPG lub JPEG lub JPG, użytkownik wybiera typ histogramów jakie chce porównywać w kontekście analizowania podobieństwa pomiędzy obrazmi:
            - Dostępne typy histogramów:
                - Histogramy RGB
                - Histogramy HOG

        - Kolejno użytkownik wybiera jedną spośród dostępnych miar niepodobieństwa:
            - Metryka Jensena-Shannona
            - Metryka Hellingera
            - Metryka Chi-kwadrat
        - Następnie użytkownik wybiera spośród załadowanych obrazów jeden plik referencyjny, dla którego chce wyszukwać podobne obrazy.
        - Aplikacja generuje podgląd pliku referencyjnego.
        - Użytkownik wybiera liczbę najbardziej podobnych obrazów do pliku referenycjnego, które ma wyszukać aplikacja.
        - Po wciśnięciu przycisku `Wyszukaj podobne obrazy`, generowana jest lista najbardziej podobnych obrazów do pliku referenycjnego, wraz z nazwą pliku, oraz wartością niepodobieństwa.
        ''')

elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa dokumentów: dane w formacie PDF":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia analizę podobieństwa dokumentów tekstowych w formacie PDF. 
        
        - Użytkownik podejmuje wybór czy chce skorzystać z własnych plików, które należy wgrać do aplikacji, czy chce wykonać test na plikach aplikacji.
        - Pliki wgrane w aplikacje obejmują statuty dziesięciu następujących organizacji:
            1. Politechnika Krakowska (PK)
            2. Akademia Górniczo Hutnicza w Krakowie (AGH)
            3. Uniwerystet Jagielloński w Krakowie (UJ)
            4. Uniwersytet Komisji Edukacji Narodowej w Krakowie (UKEN)
            5. Uniwerystet Ekonomiczny w Krakowie (UEK)
            6. Uniwersytet Rolniczy w Krakowie (UR)
            7. Bank Spółdzielczy Rzemiosła w Krakowie (BSR)
            8. Bank Spółdzielczy w Limanowej (BSL)
            9. Małopolski Bank Spółdzielczy (MBS)
            10. Bank Spółdzielczy w Brodnicy (BSB)
            11. Miejskie Przedsiębiorstwo Komunikacyjne w Krakowie (MPK)
            12. Wodociągi Miejskie w Krakowie (WMK)
        - Po wybraniu plików aplikacji, lub załadowaniu własnych plików w formacie PDF, następuje ich wgrwanie, oraz przetwarzanie na korpusy dokumentów.
        - Użytkownik wybiera następnie jedną z dostępnych miar niepodobieństwa:
            - Metryka Jensena-Shannona
            - Metryka Hellingera
            - Niepodobieństwo cosinusowe (domyślne ustawienie)
            - Niepodobieństwo $\chi^2$
        - Kolejno wyświetlana jest macierz niepodobieństwa zbioru dokumentów.
        - Użytkownik wybiera plik referenycjny, oraz liczbę najbardziej podobnych dokumentów do pliku referencyjnego (domyślne ustawienie 5).
        - Ostatecznie generowana jest lista najbardziej podobnych plików do pliku referenycjnego, wraz z wyświetleniem wartości niepodobieństwa.

        ''')

elif Opcja_pracy_w_aplikacji == "Wykrywanie wartości odstających w zbiorze danych: dane w formacie CSV/XLS/XLSX":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia wykrywanie wartości nietypowych w danych.

        - Użytkownik podejmuje wybór czy chce skorzystać z własnych plików, które należy wgrać do aplikacji, czy chce wykonać test na plikach aplikacji.
        - Pliki wgrane w aplikacje obejmują następujace zbiory danych:
            - Dane dwuwymiarowe niepochodzące z dwuwymiarowego rozkładu normalnego
            - Dane dwuwymiarowe pochodzące z dwuwymiarowego rozkładu normalnego
            - Dane trzywymiarowe niepochodzące z trzywymiarowego rozkładu normalnego
            - Dane trzywymiarowe pochodzące z trzywymiarowego rozkładu normalnego
        - Po wybraniu plików aplikacji, lub załadowaniu własnych plików w formacie CSV lub XLS lub XLSX następuje ich wgrwanie, oraz wyświetlenie podglądu wgranych danych.
        - Aplikacja zwraca informację czy dane pochodzą z rozkładu normalnego.
        - Następnie użytkownik wybiera jedną spośród dostępnych miar niepodobieństwa:
            - Jeżeli dane pochodzą z rozkładu normalnego:
                - Metryka Euklidesa
                    - Użytkwonik ustawia wartość progu odcięcia, dla którego obserwacje w zbiorze mają być identyfikowane jako odstające
                    - Aplikacja wyświetla informację o liczbie nietypowych obserwacji, oraz daje możliwość wyświetlenia szczegółowych informacji na ich temat
                    - Jeżeli użytkownik chce wyświetlić szczegółowe informacje, aplikacja generuje listę obserwacji zidentyfikowanych jako nietypowe, wskazując indeks obserwacji, oraz jej odległość od centrum danych
                    - Na koniec jeżeli dane są dwuwymiarowe, lub trzywymiarowe, aplikacja generuje wizualizację punktów danych, oraz obszaru decyzyjnego w którym znajdują się obserwacje typowe dla wybranych parametrów
                - Metryka Mahalanobisa
                    - Użytkwonik ustawia wartość parametru 1 - alpha, zgodnie z którym obserwacje w zbiorze mają być identyfikowane jako nietypowe dla modelu
                    - Aplikacja wyświetla informację o sposobie identyfikacji nietypowych obserwacji, liczbie nietypowych obserwacji, oraz daje możliwość wyświetlenia szczegółowych informacji na ich temat
                    - Jeżeli użytkownik chce wyświetlić szczegółowe informacje, aplikacja generuje listę obserwacji zidentyfikowanych jako nietypowe, wskazując indeks obserwacji, oraz jej odległość od centrum danych
                    - Na koniec jeżeli dane są dwuwymiarowe, lub trzywymiarowe, aplikacja generuje wizualizację punktów danych, oraz obszaru decyzyjnego w którym znajdują się obserwacje typowe dla wybranych parametrów

            - Jeżeli dane nie pochodzą z rozkładu normalnego:
                - Użytkownik wybiera jedną spośród dostępnych miar niepodobieństwa:
                    - Metryka Manhattan
                    - Metryka Euklidesa
                    - Metryka Czebyszewa
                - Następnie użytkwonik ustawia wartość progu odcięcia, dla którego obserwacje w zbiorze mają być identyfikowane jako odstające
                - Aplikacja wyświetla informację o liczbie nietypowych obserwacji, oraz daje możliwość wyświetlenia szczegółowych informacji na ich temat
                - Jeżeli użytkownik chce wyświetlić szczegółowe informacje, aplikacja generuje listę obserwacji zidentyfikowanych jako nietypowe, wskazując indeks obserwacji, oraz jej odległość od centrum danych
                - Na koniec jeżeli dane są dwuwymiarowe, lub trzywymiarowe, aplikacja generuje wizualizację punktów danych, oraz obszaru decyzyjnego w którym znajdują się obserwacje typowe dla wybranych parametrów

        ''')

elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa danych binarnych: dane w formacie CSV/XLS/XLSX":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia analizowanie podobieństwa pomiędzy danymi binarnymi.
                    
        - Użytkownik podejmuje wybór czy chce skorzystać z własnego pliku, który należy wgrać do aplikacji w formacie CSV lub XLS lub XLSX, czy chce wykonać test na pliku aplikacji.
        - Plik wgrany w aplikacji zawiera 250 wygenerowanych obserwacji o 10 binarnych cechach.
        - Po wybraniu pliku aplikacji, lub załadowaniu własnego pliku w formacie CSV lub XLS lub XLSX, następuje ich wgrwanie, oraz generowany jest podgląd wgranego pliku.
        - Kolejno użytkownik wybiera jedną z dostępnych miar podobieństwa lub niepodobienstwa:
            - Metryka Hamminga
            - Niepodobieństwo cosinusowe
            - Podobieństwo Tanimoto/Jaccarda
            - Podobieństwo Dice'a       
        - W kolejnym kroku użytkownik wybiera sposób prezentacji niepodobieństwa pomiędzy elementami zbioru: 
            - Wybór `Pokaż macierz niepodobieństwa`: 
                - Generuje macierz niepodobieństwa zbioru danych
            - Wybór `Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne obserwacje`: 
                - Generuje macierz niepodobieństwa zbioru danych
                - Daje możliwość wyboru obserwacji referencyjnej wraz z liczbą najbardziej podobnych obserwacji jakie aplikacja ma znaleźć 
                - Naciśnięcie przycisku `Znajdź podobne obserwacje` zwraca wiersz referencyjny, oraz macierz najbardziej podobnych obserwacji do obserwacji referencyjnej wraz z wartością niepodobieństwa i numerem wiersza w zbiorze danych.    
        ''')

elif Opcja_pracy_w_aplikacji == "Analiza niepodobieństwa Prokrustesa zbiorów punktów: dane w formacie CSV/XLS/XLSX":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia analizowanie podobieństwa pomiędzy zbiorami punktów.
                    
        - Użytkownik podejmuje wybór czy chce skorzystać z własnych plików, które należy wgrać do aplikacji w formacie CSV lub XLS lub XLSX, czy chce wykonać test na plikach aplikacji.
        - Pliki wgrane w aplikacji obejmują 14 zbiorów punktów na które składa się:
            - 10 zbiorów punktów 
            - 3 zbiory punktów podobne do zbioru o numerze 01
            - 1 zbiór punktów podobny do zbioru o numerze 04
        - Po wybraniu plików aplikacji, lub załadowaniu własnych plików w formacie CSV lub XLS lub XLSX, następuje ich wgrwanie.
        - Użytkownik wybiera sposób reprezentacji niepodobieństwa pomiędzy zbiorami:
            - Wybór `Pokaż macierz niepodobieństwa`: 
                - Generuje macierz niepodobieństwa Prokrustesa wgranych plików
            - Wybór `Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne zbiory`: 
                - Generuje macierz niepodobieństwa Prokrustesa wgranych plików
                - Daje możliwość wyboru pliku referencycjnego wraz z liczbą najbardziej podobnych zbiorów jakie aplikacja ma znaleźć 
                - Naciśnięcie przycisku `Znajdź podobne zbiory punktów` zwraca listę najbardziej podobnych plików do pliku referencyjnego wraz z wartością niepodobieństwa Prokrustesa.    
                
        ''')

elif Opcja_pracy_w_aplikacji == "Wyliczanie odległości Levenshteina pomiędzy ciągami tekstowymi":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia wyliczenie odległości Levenshteina pomiędzy dwoma ciągami tekstowymi.
                    
        - Użytkownik podaje do pól tekstowych dwa dowolne ciągi znaków.
        - Aplikacja przetwarza podane ciągi:
            1. Zamiana na małe litery
            2. Usunięcie z tekstu wszystkich znaków które nie są:
                - Literami (a-z, A-Z)
                - Cyframi (0-9) 
                - Spacjami
                - Podkreśleniami (_)
        - Użytkownik wciska przycisk `Wyświetl macierz programowania dynamicznego i oblicz odległosć Levenshteina`.
        - Ostatecznie zwracana jest macierz programowania dynamicznego ciągów zadeklarowanych przez użytkownika, oraz zwracana jest odległość Levenshteina pomiędzy nimi.
        ''')

elif Opcja_pracy_w_aplikacji == "Ilość ruchów królem na szachownicy":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwa wyliczanie minimalnej ilości potrzebnych ruchów aby dotrzeć z pola na którym znajduje się król (K) do pola docelowego (C).

        - Użytkownik wybiera oznaczenie pola na którym znajduje się król (K), oraz oznaczenie pola docelowego (C).
        - Wyświetlana jest szachownica na której zaznacze są pola na których znajduje się król (K), oraz pole docelowe (C).
        - Ostatecznie zwracana jest odległość Czebyszewa pomiędzy polami wybranymi przez użytkownika.
        ''')

elif Opcja_pracy_w_aplikacji == "Klasteryzacja i klasyfikacja danych ogólnych: dane i etykiety w formacie CSV/XLS/XLSX":
    st.subheader("Opcjonalne wyświetlenie opisu:")
    with st.expander("Opis działania modułu:"):
        st.markdown(r'''
        Aplikacja umożliwia klasteryzację danych treningowych, a następnie klasyfikację danych testowych.

        - Użytkownik podejmuje wybór czy chce skorzystać z własnych plików, które należy wgrać do aplikacji, czy chce wykonać test na plikach aplikacji.
        - Pliki wgrane w aplikacje obejmują następujace zbiory danych:
            - Dane dwuwymiarowe i etykiety
            - Dane trzywymiarowe i etykiety
            - Po wybraniu plików aplikacji, lub załadowaniu własnego pliku z danymi, oraz pliku z etykietami, w formacie CSV lub XLS lub XLSX, następuje ich wgrwanie, oraz generowany jest podgląd wgranych plików.
        - Kolejno użytkownik ustawia szereg parametrów:
            - Jądro losowo dla zachowania powtarzalności wyników
            - Maksymalną liczbę iteracji algorytmu K-Średnich
            - Tolerancję algorytmu
            - Proporcję zbiorów treningowego i testowego
        - Użytkownik w kolejnym kroku wybiera jedną spośród dostępnych miar niepodobieństwa:
            - Metryka Minkowskiego z dowolnym parametrem
            - Metryka Manhattan
            - Metryka Canberry
            - Metryka Euklidesa
            - Znormalizowana metryka Euklidesa
            - Metryka Czebyszewa
            - Kwadratowe niepodobieństwo Euklidesa
        - Po wciśnięciu przycisku aplikacja zwraca wyniki klasteryzacji zbioru treningowego i klasyfikacji zbioru testowego, oraz czas wykonania algorytmu K-Średnich.
        - Ostatecznie aplikacja generuje wizualizację podziału zbioru treningowego, oraz testowego, wraz z zaznaczeniem klastrów.

        ''')
    
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Klasyfikacja: dane i etykiety w formacie CSV/XLS/XLSX
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

if Opcja_pracy_w_aplikacji == "Klasteryzacja i klasyfikacja danych ogólnych: dane i etykiety w formacie CSV/XLS/XLSX":
    Plik_z_danymi = None
    Plik_z_etykietami = None

    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")
    Wybor_wygrywania_plikow = st.selectbox("Wybierz:",[None, "Korzystaj z plików aplikacji", "Wgraj własne dane"])

    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        Kolumna11, Kolumna21 = st.columns(2)
        with Kolumna11:
            st.subheader("Wybierz spośród dostępnych zbiorów")
            st.write("")
            with st.expander("Dostępne zbiory"):
                st.write("- Dane dwuwymiarowe")
                st.write("- Dane trzywymiarowe")
        with Kolumna21:
            st.subheader("Lista wyboru:")
            Wybor_pliku_danych = st.selectbox("Wybierz",[None, "Dane dwuwymiarowe", "Dane trzywymiarowe"])
    
    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        if Wybor_pliku_danych == "Dane dwuwymiarowe":
            Folder_pliki_Dane = r"Clustering_2D_Data"
            Folder_pliki_Etykiety = r"Clustering_2D_Labels"

            Plik_csv_Dane = next((f for f in os.listdir(Folder_pliki_Dane) if f.endswith('.csv')), None)
            Plik_csv_Etykiety = next((f for f in os.listdir(Folder_pliki_Etykiety) if f.endswith('.csv')), None)
            
            if Plik_csv_Dane:
                Sciezka_pliku = os.path.join(Folder_pliki_Dane, Plik_csv_Dane)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv_Dane

            if Plik_csv_Etykiety:
                Sciezka_pliku = os.path.join(Folder_pliki_Etykiety, Plik_csv_Etykiety)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_etykietami = BytesIO(f.read())
                    Plik_z_etykietami.name = Plik_csv_Etykiety

        elif Wybor_pliku_danych == "Dane trzywymiarowe":
            Folder_pliki_Dane = r"Clustering_3D_Data"
            Folder_pliki_Etykiety = r"Clustering_3D_Labels"

            Plik_csv_Dane = next((f for f in os.listdir(Folder_pliki_Dane) if f.endswith('.csv')), None)
            Plik_csv_Etykiety = next((f for f in os.listdir(Folder_pliki_Etykiety) if f.endswith('.csv')), None)
            
            if Plik_csv_Dane:
                Sciezka_pliku = os.path.join(Folder_pliki_Dane, Plik_csv_Dane)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv_Dane

            if Plik_csv_Etykiety:
                Sciezka_pliku = os.path.join(Folder_pliki_Etykiety, Plik_csv_Etykiety)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_etykietami = BytesIO(f.read())
                    Plik_z_etykietami.name = Plik_csv_Etykiety


    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        st.subheader("Wczytaj dane oraz etykiety w formacie CSV/XLS/XLSX")
        Plik_z_danymi = st.file_uploader("Wgraj plik z danymi w formacie CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])
        Plik_z_etykietami = st.file_uploader("Wgraj plik z etykietami CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])

    if Plik_z_danymi != None and Plik_z_etykietami != None:
        if Plik_z_danymi.name.endswith(('xls', 'xlsx')):
            Dane = pd.read_excel(Plik_z_danymi)
        else:
            Dane = pd.read_csv(Plik_z_danymi)

        if Plik_z_etykietami.name.endswith(('xls', 'xlsx')):
            Etykiety = pd.read_excel(Plik_z_etykietami)
        else:
            Etykiety = pd.read_csv(Plik_z_etykietami)

        st.subheader("Rozwiń aby podejrzeć wczytane pliki")
        Kolumna111, Kolumna112 = st.columns(2)
        with Kolumna111:
            with st.expander("Podgląd zbioru danych:"):
                st.dataframe(Dane)
        with Kolumna112:
            with st.expander("Podgląd etykiet:"):
                st.dataframe(Etykiety)

        Dane = Dane.to_numpy()
        Etykiety = Etykiety.to_numpy()

        Kolumna1P, Kolumna2P, Kolumna3P, Kolumna4P = st.columns(4)
        
        with Kolumna1P:
            st.subheader("Ustaw jądro losowe algorytmu")
            Jadro_losowe_uzytkownika = st.slider(
                "Wybierz jądro losowe dla powtarzalności wyników:",
                min_value=0,
                max_value=100,
                value=42,
                step=1
            )
        with Kolumna2P:
            st.subheader("Ustaw maksymalną liczbę iteracji")
            Maksymalna_ilosc_iteracji = st.slider(
                "Wybierz maksymalną ilość iteracji algorytmu:",
                min_value=0,
                max_value=1000,
                value=100, 
                step=1
            )
        with Kolumna3P:
            st.subheader("Wybierz tolerancję algorytmu")
            Opcje_tolerancji = [1e-14,1e-13,1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            Opcje_tolerancji_formatowane = [f"{val:.0e}" for val in Opcje_tolerancji]
            wybrana_tolerancja = st.selectbox(
                "Wybierz jedną z dostępnych wartości tolerancji:",
                Opcje_tolerancji_formatowane
            )
        Tolerancja = float(wybrana_tolerancja)
        with Kolumna4P:
            st.subheader("Ustaw proporcję zbioru testowego")
            Proporcja_testowy = st.slider(
                "Wybierz proporcję zbioru testowego (w %):",
                min_value=0,
                max_value=100,
                value=20, 
                step=1
            )
        Proporcja_treningowy = 100 - Proporcja_testowy

        Proporcja_zbioru_testowego = Proporcja_testowy / 100
        Proporcja_zbioru_treningowego = Proporcja_treningowy / 100

        Kolumna1M, Kolumna2M = st.columns(2)
        with Kolumna1M:
            st.subheader("Wybierz miarę niepodobieństwa")
            st.write("")
            with st.expander("Dostępne miary"):
                st.write("- Metryka Minkowskiego z dowolnym parametrem")
                st.write("- Metryka Manhattan")
                st.write("- Metryka Canberry")
                st.write("- Metryka Euklidesa")
                st.write("- Znormalizowana metryka Euklidesa")
                st.write("- Metryka Czebyszewa")
                st.write("- Kwadratowe niepodobieństwo Euklidesa")

        with Kolumna2M:
            st.subheader("Lista wyboru:")
            Wybor_miary = st.selectbox("Wybierz:",
                [   "Metryka Minkowskiego z dowolnym parametrem",
                    "Metryka Manhattan",
                    "Metryka Canberry",
                    "Metryka Euklidesa",
                    "Znormalizowana metryka Euklidesa",
                    "Metryka Czebyszewa",
                    "Kwadratowe niepodobieństwo Euklidesa",
                ]
            )

        if Wybor_miary == "Metryka Minkowskiego z dowolnym parametrem":
            st.subheader("Ustaw wartość parametru przed kontynuacją")
            Parametr_p = st.number_input("Wybierz parametr metryki Minkowskiego:", min_value=1.0, max_value=1000000.0, value=2.0)


        st.write(f"Wybrana miara: **{Wybor_miary}**")


        if st.button("Rozpocznij klasyfikację"):
            st.write(f"Rozpoczynam klasyfikację z wybraną miarą: {Wybor_miary}")

            if Wybor_miary == "Metryka Euklidesa":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich(Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Odleglosc_Euklidesa,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Odleglosc_Euklidesa)

                    Fig_testowe = Wizualizacja_klastrow(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Odleglosc_Euklidesa)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)

            elif Wybor_miary == "Metryka Minkowskiego z dowolnym parametrem":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich_Minkowski(Dane,
                    Etykiety,
                    Parametr_p,
                    float(Proporcja_zbioru_testowego),
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow_Minkowski(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Parametr_p)

                    Fig_testowe = Wizualizacja_klastrow_Minkowski(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Parametr_p)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)

            elif Wybor_miary == "Metryka Manhattan":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich(Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Odleglosc_Manhattan,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Odleglosc_Manhattan)

                    Fig_testowe = Wizualizacja_klastrow(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Odleglosc_Manhattan)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)

            elif Wybor_miary == "Metryka Canberry":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich(Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Odleglosc_Canberry,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")


                Fig_treningowe = Wizualizacja_klastrow(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Odleglosc_Canberry)

                Fig_testowe = Wizualizacja_klastrow(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Odleglosc_Canberry)

                Kolumna1W, Kolumna2W = st.columns(2)
                with Kolumna1W:
                    st.subheader("Wizualizacja dla zbioru treningowego")
                    st.plotly_chart(Fig_treningowe, use_container_width=False)
                with Kolumna2W:
                    st.subheader("Wizualizacja dla zbioru testowego")
                    st.plotly_chart(Fig_testowe, use_container_width=False)

            elif Wybor_miary == "Metryka Czebyszewa":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich(Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Odleglosc_Czebyszewa,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Odleglosc_Czebyszewa)

                    Fig_testowe = Wizualizacja_klastrow(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Odleglosc_Czebyszewa)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)

            elif Wybor_miary == "Kwadratowe niepodobieństwo Euklidesa":
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich(Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Kwadratowe_niepodobienstwo_Euklidesa,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa = Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym:{Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Kwadratowe_niepodobienstwo_Euklidesa)

                    Fig_testowe = Wizualizacja_klastrow(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Kwadratowe_niepodobienstwo_Euklidesa)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)


            elif Wybor_miary == "Znormalizowana metryka Euklidesa":
                Wektor_odchylen_standardowych = np.std(Dane, axis=0, ddof=1)
                Wektor_odchylen_standardowych = Wektor_odchylen_standardowych.reshape(1,-1)
                #st.write("Ochylenie standardowe cech w zbiorze:")
                #st.dataframe(pd.DataFrame(Wektor_odchylen_standardowych))
                Wagi = 1/Wektor_odchylen_standardowych**2
                st.subheader("Uzyskane wyniki")
                Zbior_danych_treningowych, Zbior_danych_testowych, Etykiety_treningowe, Etykiety_testowe, Centroidy, Dokladnosc_treningowa, Dokladnosc_testowa, Czas_wykonywania = Algorytm_K_Srednich_Znormalizowana(
                    Dane,
                    Etykiety,
                    float(Proporcja_zbioru_testowego),
                    Wazona_odleglosc_Euklidesa,
                    Wagi,
                    Maksymalna_ilosc_iteracji,
                    Tolerancja,
                    Liczba_losowa=Jadro_losowe_uzytkownika)
                Ilosc_klastrow = len(np.unique(Etykiety))
                st.success(f"Dokładność klasyfikacji na zbiorze treningowym: {Dokladnosc_treningowa}")
                st.success(f"Dokładność klasyfikacji na zbiorze testowym: {Dokladnosc_testowa}")
                st.success(f"Czas wykonania algorytmu: {Czas_wykonywania}")

                with st.spinner("Generowanie wizualizacji"):
                    Fig_treningowe = Wizualizacja_klastrow_Znormalizowana(Zbior_danych_treningowych, Etykiety_treningowe, Centroidy, Ilosc_klastrow, Wagi)

                    Fig_testowe = Wizualizacja_klastrow_Znormalizowana(Zbior_danych_testowych, Etykiety_testowe, Centroidy, Ilosc_klastrow, Wagi)

                    Kolumna1W, Kolumna2W = st.columns(2)
                    with Kolumna1W:
                        st.subheader("Wizualizacja dla zbioru treningowego")
                        st.plotly_chart(Fig_treningowe, use_container_width=False)
                    with Kolumna2W:
                        st.subheader("Wizualizacja dla zbioru testowego")
                        st.plotly_chart(Fig_testowe, use_container_width=False)


###################################################################################################################################
###################################################################################################################################                
###################################################################################################################################
# Klasyfikacja: Analiza podobieństwa: dane w formacie PDF
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa dokumentów: dane w formacie PDF":

    if "Pliki_do_porownania" not in st.session_state:
        st.session_state.Pliki_do_porownania = None
        st.session_state.Zbior_nazw_plikow = None
        st.session_state.Zbior_przetworzonych_plikow = None
        st.session_state.Macierz_podobienstwa = None
        st.session_state.Wybor_miary = None
        st.session_state.Wybor_miary_poprzednia = None

    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")
    
    if "Poprzednia_opcja" not in st.session_state:
        st.session_state.Poprzednia_opcja = None

    Wybor_wygrywania_plikow = st.selectbox("Wybierz:", [None, "Korzystaj z plików aplikacji", "Wgraj własne dane"])

    if Wybor_wygrywania_plikow != st.session_state.Poprzednia_opcja:
        st.session_state.Pliki_do_porownania = None
        st.session_state.Zbior_nazw_plikow = None
        st.session_state.Zbior_przetworzonych_plikow = None
        st.session_state.Macierz_podobienstwa = None
        st.session_state.Wybor_miary = None
        st.session_state.Poprzednia_opcja = Wybor_wygrywania_plikow

    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        if st.session_state.Pliki_do_porownania is None:
            with st.spinner("Wgrywanie plików aplikacji..."):
                Folder_pliki = r"Statuty_organizacji"
                Pliki_pdf = [f for f in os.listdir(Folder_pliki) if f.endswith('.pdf')]

                if Pliki_pdf:
                    Pliki_do_porownania = []
                    for Plik in Pliki_pdf:
                        Sciezka_pliku = os.path.join(Folder_pliki, Plik)
                        with open(Sciezka_pliku, "rb") as f:
                            file_bytes = BytesIO(f.read())
                            file_bytes.name = Plik
                            Pliki_do_porownania.append(file_bytes)
                    st.session_state.Pliki_do_porownania = Pliki_do_porownania

    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        Zaczytane_pliki = st.file_uploader(
            "Wgraj pliki w formacie PDF",
            type=["pdf"],
            accept_multiple_files=True
        )

        if Zaczytane_pliki and Zaczytane_pliki != st.session_state.Pliki_do_porownania:
            st.session_state.Pliki_do_porownania = Zaczytane_pliki
            st.session_state.Zbior_przetworzonych_plikow = None

    if st.session_state.Pliki_do_porownania is None or len(st.session_state.Pliki_do_porownania) == 0:
        st.stop()


    if st.session_state.Zbior_przetworzonych_plikow is None:
        with st.spinner("Przetwarzanie wgranych plików..."):
            Zbior_nazw_plikow = []
            Zbior_przetworzonych_plikow = []

            for Plik in st.session_state.Pliki_do_porownania:
                Tekst_z_pdf = Uzyskaj_tekst_z_PDF(Plik)
                Tekst_przetworzony = Przygotuj_tekst_do_analizy(Tekst_z_pdf)
                Zbior_nazw_plikow.append(Plik.name)
                Zbior_przetworzonych_plikow.append(Tekst_przetworzony)

            st.session_state.Zbior_nazw_plikow = Zbior_nazw_plikow
            st.session_state.Zbior_przetworzonych_plikow = Zbior_przetworzonych_plikow

    st.success("Pliki zostały poprawnie przetworzone.")

    if st.session_state.Zbior_przetworzonych_plikow is not None:
        Wektoryzer = CountVectorizer()
        Zamiana_na_korpusy = Wektoryzer.fit_transform(st.session_state.Zbior_przetworzonych_plikow).toarray()
        st.subheader("Pliki zamienione na korpusy:")
        with st.expander("Zobacz pogląd:"):
            st.dataframe(pd.DataFrame(Zamiana_na_korpusy))

        st.subheader("Wybierz miarę niepodobieństwa")
        with st.expander("Dostępne miary"):
            st.write("- Niepodobieństwo cosinusowe")
            st.write("- Metryka Jensena-Shannona")
            st.write("- Metryka Hellingera")
            st.write("- Niepodobieństwo chi-kwadrat")


        Wybor_miary = st.selectbox(
            "Wybierz:",
            ["Niepodobieństwo cosinusowe", "Metryka Jensena-Shannona", "Metryka Hellingera", "Niepodobieństwo chi-kwadrat"],
            key="Wybor_miary"
        )

        if Wybor_miary != st.session_state.Wybor_miary_poprzednia:
            st.session_state.Macierz_podobienstwa = None
            st.session_state.Wybor_miary_poprzednia = Wybor_miary

        if st.session_state.Macierz_podobienstwa is None and Wybor_miary:
            Ilosc_plikow = Zamiana_na_korpusy.shape[0]
            Macierz_podobienstwa = np.zeros((Ilosc_plikow, Ilosc_plikow))

            with st.spinner("Obliczanie macierzy niepodobieństwa..."):
                for i in range(Ilosc_plikow):
                    for j in range(i, Ilosc_plikow):
                        try:
                            if Wybor_miary == "Metryka Jensena-Shannona":
                                podobienstwo = Odleglosc_Jensena_Shannona(Zamiana_na_korpusy[i], Zamiana_na_korpusy[j])
                            elif Wybor_miary == "Metryka Hellingera":
                                podobienstwo = Odleglosc_Hellingera(Zamiana_na_korpusy[i], Zamiana_na_korpusy[j])
                            elif Wybor_miary == "Niepodobieństwo cosinusowe":
                                podobienstwo = Niepodobienstwo_cosinusowe(Zamiana_na_korpusy[i], Zamiana_na_korpusy[j])
                            elif Wybor_miary == "Niepodobieństwo chi-kwadrat":
                                podobienstwo = Niepodobienstwo_chi_kwadrat(Zamiana_na_korpusy[i], Zamiana_na_korpusy[j])
                            else:
                                podobienstwo = 0
                            
                            Macierz_podobienstwa[i, j] = podobienstwo
                            Macierz_podobienstwa[j, i] = podobienstwo
                        except Exception as e:
                            st.error(f"Błąd obliczeń dla pary ({i}, {j}): {e}")

                st.session_state.Macierz_podobienstwa = Macierz_podobienstwa

        if st.session_state.Macierz_podobienstwa is not None:
            st.subheader("Macierz niepodobieństwa:")
            with st.expander("Wyświetl macierz niepodobieństwa"):
                st.dataframe(pd.DataFrame(
                    st.session_state.Macierz_podobienstwa, 
                    index=st.session_state.Zbior_nazw_plikow, 
                    columns=st.session_state.Zbior_nazw_plikow
                ))

            Kolumna1T, Kolumna2T, Kolumna3T = st.columns(3)

            with Kolumna1T:
                st.subheader("Wybierz plik referencyjny")
                Wybor_pliku = st.selectbox("Wybierz:", st.session_state.Zbior_nazw_plikow)
            with Kolumna2T:
                st.subheader("Wybierz liczbę podobnych plików")
                Liczba_podobnych = st.slider(
                    "Liczba najbardziej podobnych plików:", 
                    min_value=1, 
                    max_value=len(st.session_state.Zbior_nazw_plikow) - 1, 
                    value=5
                )

            if Wybor_pliku:
                Index_pliku = st.session_state.Zbior_nazw_plikow.index(Wybor_pliku)
                Podobienstwa = st.session_state.Macierz_podobienstwa[Index_pliku]
                Indeksy_podobnych = np.argsort(Podobienstwa)[1:Liczba_podobnych + 1]
                with Kolumna3T:
                    st.subheader("Najbardziej podobne pliki:")
                    for i, idx in enumerate(Indeksy_podobnych):
                        st.write(f"{i + 1}. {st.session_state.Zbior_nazw_plikow[idx]} (Niepodobieństwo: {Podobienstwa[idx]:.4f})")

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Klasyfikacja: dane w formacie JPG/JPEG/PNG i etykiety w formacie CSV
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa obrazów: dane formacie JPG/JPEG/PNG":

    if "Pliki_MNIST" not in st.session_state:
        st.session_state.Pliki_MNIST = None
        st.session_state.Pliki_TS = None
        st.session_state.Histogramy_RGBHOG_MNIST = None
        st.session_state.Histogramy_RGBHOG_TS = None

    with st.spinner("Ładowanie plików aplikacji..."):
        Folder_pliki_MNIST = r"MNIST_DIGITS"
        Folder_pliki_TS = r"ZNAKI_DROGOWE"

        if st.session_state.Pliki_MNIST is None:
            Pliki_png_MNIST = [f for f in os.listdir(Folder_pliki_MNIST) if f.endswith('.png')]
            Zaczytane_pliki_MNIST = []
            for Plik in Pliki_png_MNIST:
                Sciezka_pliku = os.path.join(Folder_pliki_MNIST, Plik)
                with open(Sciezka_pliku, "rb") as f:
                    file_bytes = BytesIO(f.read())
                    file_bytes.name = Plik
                    Zaczytane_pliki_MNIST.append(file_bytes)
            st.session_state.Pliki_MNIST = Zaczytane_pliki_MNIST

        if st.session_state.Pliki_TS is None:
            Pliki_png_TS = [f for f in os.listdir(Folder_pliki_TS) if f.endswith('.png')]
            Zaczytane_pliki_TS = []
            for Plik in Pliki_png_TS:
                Sciezka_pliku = os.path.join(Folder_pliki_TS, Plik)
                with open(Sciezka_pliku, "rb") as f:
                    file_bytes = BytesIO(f.read())
                    file_bytes.name = Plik
                    Zaczytane_pliki_TS.append(file_bytes)
            st.session_state.Pliki_TS = Zaczytane_pliki_TS

        st.success("Wczytano poprawnie wszystkie pliki aplikacji")

    with st.spinner("Przetwarzanie zaczytanych plików..."):
        if st.session_state.Histogramy_RGBHOG_MNIST is None:
            st.session_state.Histogramy_RGBHOG_MNIST = Przetwarzaj_histogramy(st.session_state.Pliki_MNIST)
        if st.session_state.Histogramy_RGBHOG_TS is None:
            st.session_state.Histogramy_RGBHOG_TS = Przetwarzaj_histogramy(st.session_state.Pliki_TS)

        Histogramy_RGBHOG_MNIST = st.session_state.Histogramy_RGBHOG_MNIST
        Histogramy_RGBHOG_TS = st.session_state.Histogramy_RGBHOG_TS

        Histogramy_RGB_MNIST, Histogramy_HOG_MNIST = zip(*Histogramy_RGBHOG_MNIST)
        Histogramy_RGB_TS, Histogramy_HOG_TS = zip(*Histogramy_RGBHOG_TS)

        st.success("Wszystkie pliki zostały poprawnie przetworzone")

    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")
    Wybor_wygrywania_plikow = st.selectbox("Wybierz:",
                                    [None, "Korzystaj z plików aplikacji", "Wgraj własne dane"])
    
    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        st.subheader("Wybierz spośród dostępnych zbiorów")
        Wybor_pliku_danych = st.selectbox("Wybierz:",[None, "Cyfry MNIST","Znaki drogowe"])

        Kolumna1111, Kolumna1112 = st.columns(2)

        if Wybor_pliku_danych != None:
            with Kolumna1111:
                st.subheader("Wybierz typ histogramów które chcesz porównywać")
                Wybor_typu_histogramu = st.selectbox("Wybierz:",[None, "Histogramy RGB", "Histogramy HOG"])

            if Wybor_typu_histogramu != None:
                with Kolumna1112:
                    st.subheader("Wybierz miarę niepodobieństwa")
                    Miara_niepodobienstwa = st.selectbox("Wybierz:", [None, "Metryka Jensena-Shannona", "Metryka Hellingera", "Niepodobieństwo Chi-kwadrat"])

                Kolumna1O, Kolumna2O = st.columns(2)

                if Miara_niepodobienstwa != None:
                    with Kolumna1O:
                        st.subheader("Wybierz plik referencyjny")
                        if Wybor_pliku_danych == "Cyfry MNIST":
                            Wybrany_plik = st.selectbox("Wybierz plik referencyjny:",[Plik.name for Plik in st.session_state.Pliki_MNIST])
                            Wybrany_indeks = [Plik.name for Plik in st.session_state.Pliki_MNIST].index(Wybrany_plik)
                            with st.expander("Podgląd wybranego pliku:"):
                                st.image(st.session_state.Pliki_MNIST[Wybrany_indeks], caption=f"Wybrany plik: {Wybrany_plik}", use_container_width=True)

                            Liczba_podobnych = st.slider("Wybierz liczbę najbardziej podobnych obrazów:",
                            min_value=1,
                            max_value=len(st.session_state.Pliki_MNIST) - 1,
                            value=5)

                        elif Wybor_pliku_danych == "Znaki drogowe":
                            Wybrany_plik = st.selectbox("Wybierz plik referencyjny:",[Plik.name for Plik in st.session_state.Pliki_TS])
                            Wybrany_indeks = [Plik.name for Plik in st.session_state.Pliki_TS].index(Wybrany_plik)
                            with st.expander("Podgląd wybranego pliku:"):
                                st.image(st.session_state.Pliki_TS[Wybrany_indeks], caption=f"Wybrany plik: {Wybrany_plik}", use_container_width=False, width=400)

                            st.subheader("Wybierz liczbę podobnych obrazów")
                            Liczba_podobnych = st.slider("Wybierz liczbę najbardziej podobnych obrazów:",
                                min_value=1,
                                max_value=len(st.session_state.Pliki_TS) - 1,
                                value=5)
                    with Kolumna2O:
                        st.subheader("Najbardziej podobne obrazy")
                        if st.button("Wyszukaj podobne obrazy"):
                            if Wybor_pliku_danych == "Cyfry MNIST" and Wybor_typu_histogramu == "Histogramy RGB":
                                Zaczytane_pliki = st.session_state.Pliki_MNIST
                                Wybrany_histogram = Histogramy_RGB_MNIST[Wybrany_indeks]
                                Histogramy = Histogramy_RGB_MNIST

                            elif Wybor_pliku_danych == "Cyfry MNIST" and Wybor_typu_histogramu == "Histogramy HOG":
                                Zaczytane_pliki = st.session_state.Pliki_MNIST
                                Wybrany_histogram = Histogramy_HOG_MNIST[Wybrany_indeks]
                                Histogramy = Histogramy_HOG_MNIST
                            elif Wybor_pliku_danych == "Znaki drogowe" and Wybor_typu_histogramu == "Histogramy RGB":
                                Zaczytane_pliki = st.session_state.Pliki_TS
                                Wybrany_histogram = Histogramy_RGB_TS[Wybrany_indeks]
                                Histogramy = Histogramy_RGB_TS
                            elif Wybor_pliku_danych == "Znaki drogowe" and Wybor_typu_histogramu == "Histogramy HOG":
                                Zaczytane_pliki = st.session_state.Pliki_TS
                                Wybrany_histogram = Histogramy_HOG_TS[Wybrany_indeks]
                                Histogramy = Histogramy_HOG_TS

                            Odleglosci = []
                            for i, Histogram in enumerate(Histogramy):
                                if i != Wybrany_indeks:
                                    if Miara_niepodobienstwa == "Niepodobieństwo Chi-kwadrat":
                                        Odleglosc = Niepodobienstwo_chi_kwadrat(Wybrany_histogram, Histogram)
                                    elif Miara_niepodobienstwa == "Metryka Hellingera":
                                        Odleglosc = Odleglosc_Hellingera(Wybrany_histogram, Histogram)
                                    elif Miara_niepodobienstwa == "Metryka Jensena-Shannona":
                                        Odleglosc = Odleglosc_Jensena_Shannona(Wybrany_histogram, Histogram)
                                    Odleglosci.append((i, Odleglosc))

                            Odleglosci = sorted(Odleglosci, key=lambda x: x[1])

                            st.write("Najbardziej podobne obrazy:")
                            for indeks, odleglosc in Odleglosci[:Liczba_podobnych]:
                                st.image(Zaczytane_pliki[indeks], caption=f"Plik: {Zaczytane_pliki[indeks].name}, Niepodobieństwo: {odleglosc:.2f}")


    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        st.subheader("Wgraj pliki w formacie JPG/JPEG/PNG")

        if "Wgrane_pliki" not in st.session_state:
            st.session_state.Wgrane_pliki = None
            st.session_state.Histogramy_RGB = None
            st.session_state.Histogramy_HOG = None

        Zaczytane_pliki = st.file_uploader(
            "Wgraj pliki w formacie JPG/JPEG/PNG",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )

        if Zaczytane_pliki:
            if st.session_state.Wgrane_pliki != Zaczytane_pliki:
                with st.spinner("Przetwarzanie plików na histogramy..."):
                    Histogramy_przetwarzanie = Parallel(n_jobs=-1)(
                        delayed(lambda plik: (
                            Histogram_kolorow_RGB_dla_obrazu(np.array(Image.open(plik))),
                            Histogram_HOG_dla_obrazu(np.array(Image.open(plik)))
                        ))(plik) for plik in Zaczytane_pliki
                    )
                    st.session_state.Wgrane_pliki = Zaczytane_pliki
                    st.session_state.Histogramy_RGB, st.session_state.Histogramy_HOG = zip(*Histogramy_przetwarzanie)

                st.success("Wszystkie pliki zostały poprawnie przetworzone")

            Histogramy_RGB = st.session_state.Histogramy_RGB
            Histogramy_HOG = st.session_state.Histogramy_HOG

            st.subheader("Wybierz typ histogramów które chcesz porównywać")
            Wybor_typu_histogramu = st.selectbox("Wybierz:", [None, "Histogramy RGB", "Histogramy HOG"])

            if Wybor_typu_histogramu != None:
                st.subheader("Wybierz miarę niepodobieństwa")
                Miara_niepodobienstwa = st.selectbox("Wybierz", [None, "Metryka Jensena-Shannona", "Metryka Hellingera", "Niepodobieństwo Chi-kwadrat"])

                if Miara_niepodobienstwa != None:
                    st.subheader("Wybierz plik referencyjny")
                    Wybrany_plik = st.selectbox("Wybierz plik referencyjny:", [plik.name for plik in st.session_state.Wgrane_pliki])
                    Wybrany_indeks = [plik.name for plik in st.session_state.Wgrane_pliki].index(Wybrany_plik)
                    st.image(st.session_state.Wgrane_pliki[Wybrany_indeks], caption=f"Wybrany plik: {Wybrany_plik}", use_container_width=True)

                    Liczba_podobnych = st.slider(
                        "Wybierz liczbę najbardziej podobnych obrazów:",
                        min_value=1,
                        max_value=len(st.session_state.Wgrane_pliki) - 1,
                        value=5
                    )

                    if st.button("Wyszukaj podobne obrazy"):
                        if Wybor_typu_histogramu == "Histogramy RGB":
                            Wybrany_histogram = Histogramy_RGB[Wybrany_indeks]
                            Histogramy = Histogramy_RGB
                        elif Wybor_typu_histogramu == "Histogramy HOG":
                            Wybrany_histogram = Histogramy_HOG[Wybrany_indeks]
                            Histogramy = Histogramy_HOG

                        Odleglosci = []
                        for i, Histogram in enumerate(Histogramy):
                            if i != Wybrany_indeks:
                                if Miara_niepodobienstwa == "Niepodobieństwo Chi-kwadrat":
                                    Odleglosc = Niepodobienstwo_chi_kwadrat(Wybrany_histogram, Histogram)
                                elif Miara_niepodobienstwa == "Metryka Hellingera":
                                    Odleglosc = Odleglosc_Hellingera(Wybrany_histogram, Histogram)
                                elif Miara_niepodobienstwa == "Metryka Jensena-Shannona":
                                    Odleglosc = Odleglosc_Jensena_Shannona(Wybrany_histogram, Histogram)
                                Odleglosci.append((i, Odleglosc))

                        Odleglosci = sorted(Odleglosci, key=lambda x: x[1])

                        st.write("Najbardziej podobne obrazy:")
                        for indeks, odleglosc in Odleglosci[:Liczba_podobnych]:
                            st.image(st.session_state.Wgrane_pliki[indeks], caption=f"Plik: {st.session_state.Wgrane_pliki[indeks].name}, Odległość: {odleglosc:.2f}")



###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Analiza podobieństwa danych binarnych
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

elif Opcja_pracy_w_aplikacji == "Analiza podobieństwa danych binarnych: dane w formacie CSV/XLS/XLSX":
    Plik_z_danymi = None
    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")

    Wybor_wygrywania_plikow = st.selectbox("Wybierz:",
                                    [None, "Korzystaj z plików aplikacji","Wgraj własne dane"])
    
    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        Folder_pliki = r"Binarne_dane"

        Plik_csv = next((f for f in os.listdir(Folder_pliki) if f.endswith('.csv')), None)
        
        if Plik_csv:
            Sciezka_pliku = os.path.join(Folder_pliki, Plik_csv)
            with open(Sciezka_pliku, "rb") as f:
                Plik_z_danymi = BytesIO(f.read())
                Plik_z_danymi.name = Plik_csv


    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        st.subheader("Wgraj pliki w formacie CSV/XLS/XLSX")
        Plik_z_danymi = st.file_uploader("Wgraj plik z danymi w formacie CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])

    

    if Plik_z_danymi != None:
        if Plik_z_danymi.name.endswith(('xls', 'xlsx')):
            Dane = pd.read_excel(Plik_z_danymi)
        else:
            Dane = pd.read_csv(Plik_z_danymi)

        st.subheader("Opcjonalny podgląd zbioru danych:")
        with st.expander("Wyświetl załadowany zbiór:"):
            st.dataframe(Dane)

        Dane = Dane.to_numpy()

        Kolumna1B, Kolumna2B, Kolumna3B = st.columns(3)

        with Kolumna1B:
            st.subheader("Wybierz miarę podobieństwa lub niepodobieństwa")

            with st.expander("Miary dostępne na liście:"):
                st.write("- Metryka Hamminga")
                st.write("- Niepodobieństwo cosinusowe")
                st.write("- Podobieństwo Tanimoto/Jaccarada")
                st.write("- Podobieństwo Dice'a")
        with Kolumna2B:
            st.subheader("Lista wyboru")
            Wybor_miary = st.selectbox("Wybierz:",[None, "Metryka Hamminga", "Niepodobieństwo cosinusowe", "Podobieństwo Tanimoto/Jaccarda", "Podobieństwo Dice'a",])

        Ilosc_wierszy = Dane.shape[0]
        Macierz_niepodobienstwa = np.zeros((Ilosc_wierszy, Ilosc_wierszy))
        for i in range(Ilosc_wierszy):
            for j in range(i, Ilosc_wierszy):
                if Wybor_miary == "Metryka Hamminga":
                    Odleglosc = Odleglosc_Hamminga(Dane[i], Dane[j])
                    Macierz_niepodobienstwa[i, j] = Odleglosc
                    Macierz_niepodobienstwa[j, i] = Odleglosc
                elif Wybor_miary == "Niepodobieństwo cosinusowe":
                    Odleglosc = Niepodobienstwo_cosinusowe(Dane[i], Dane[j])
                    Macierz_niepodobienstwa[i, j] = Odleglosc
                    Macierz_niepodobienstwa[j, i] = Odleglosc
                elif Wybor_miary == "Podobieństwo Tanimoto/Jaccarda":
                    Odleglosc = Niepodobienstwo_Jaccarda(Dane[i], Dane[j])
                    Macierz_niepodobienstwa[i, j] = Odleglosc
                    Macierz_niepodobienstwa[j, i] = Odleglosc
                elif Wybor_miary == "Podobieństwo Dice'a":
                    Odleglosc = Niepodobienstwo_Dicea(Dane[i], Dane[j])
                    Macierz_niepodobienstwa[i, j] = Odleglosc
                    Macierz_niepodobienstwa[j, i] = Odleglosc
                    
        with Kolumna3B:
            st.subheader("Wybierz opcję wyświetlania")
            Opcja_wyswietlania = st.radio(
                "Wybierz sposób prezentacji:",
                ["Pokaż macierz niepodobieństwa", "Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne obserwacje"]
            )

        if Opcja_wyswietlania == "Pokaż macierz niepodobieństwa":
            st.subheader("Macierz niepodobieństwa zbioru danych")
            st.dataframe(pd.DataFrame(Macierz_niepodobienstwa))

        elif Opcja_wyswietlania == "Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne obserwacje":
            st.subheader("Macierz niepodobieństwa zbioru danych")
            with st.expander("Wyświetl macierz niepodobieństwa"):
                st.dataframe(pd.DataFrame(Macierz_niepodobienstwa))

            Kolumna12B, Kolumna22B = st.columns(2)
            with Kolumna12B:
                st.subheader("Podaj numer obserwacji referencyjnej")
                Numer_wiersza = st.number_input("Podaj numer:", min_value=0, max_value=Ilosc_wierszy-1, step=1)
            with Kolumna22B:
                st.subheader("Ustaw liczbę podobnych obserwacji")
                Liczba_podobnych = st.number_input("Podaj liczbę najbardziej podobnych obserwacji do odszukania:", min_value=1, max_value=Ilosc_wierszy-1, step=1)

            if st.button("Znajdź podobne obserwacje"):
                Niepodobienstwa = Macierz_niepodobienstwa[Numer_wiersza]
                Indeksy_podobnych = np.argsort(Niepodobienstwa)[1:int(Liczba_podobnych)+1]

                st.subheader("Obserwacja referencyjna:")
                Wiersz_referencyjny = pd.DataFrame([Dane[Numer_wiersza]], columns=[f"Feature_{i+1}" for i in range(Dane.shape[1])])
                Wiersz_referencyjny.insert(0, "Index", [Numer_wiersza])
                st.dataframe(Wiersz_referencyjny)

                st.subheader("Najbardziej podobne obserwacje:")
                Podobne_wiersze = pd.DataFrame(Dane[Indeksy_podobnych], columns=[f"Feature_{i+1}" for i in range(Dane.shape[1])])
                Podobne_wiersze.insert(0, "Niepodobieństwo", Niepodobienstwa[Indeksy_podobnych])
                Podobne_wiersze.insert(1, "Index", Indeksy_podobnych)
                st.dataframe(Podobne_wiersze)

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Analiza podobieństwa punktów w przestrzeni: dane w formacie CSV
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

elif Opcja_pracy_w_aplikacji == "Analiza niepodobieństwa Prokrustesa zbiorów punktów: dane w formacie CSV/XLS/XLSX":

    Kolumna1P, Kolumna2P = st.columns(2)

    Pliki_do_porownania = None

    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")

    Wybor_wygrywania_plikow = st.selectbox("Wybierz:",
                                    [None, "Korzystaj z plików aplikacji","Wgraj własne dane"])

    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        Folder_pliki = r"Prokrustes_datasets"
    
        Pliki_do_porownania = []
        for nazwa_pliku in os.listdir(Folder_pliki):
            if nazwa_pliku.endswith('.csv'):
                sciezka = os.path.join(Folder_pliki, nazwa_pliku)
                with open(sciezka, "rb") as f:
                    plik_bytes = BytesIO(f.read())
                    plik_bytes.name = nazwa_pliku
                    Pliki_do_porownania.append(plik_bytes)

    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        st.subheader("Wgraj pliki w formie CSV/XLS/XLSX")


        Pliki_do_porownania = st.file_uploader(
            "Wgraj pliki w formacie CSV/XLS/XLSX",
            type=["csv","xls","xlsx"],
            accept_multiple_files=True
        )

    if Pliki_do_porownania != None:
        Zbior_zbiorow = []
        Nazwy_plikow = []

        for Plik in Pliki_do_porownania:
            if Plik.name.endswith(('xls','xlsx')):
                Dane = pd.read_excel(Plik)
            else:
                Dane = pd.read_csv(Plik)
            Dane = Dane.to_numpy()
            Zbior_zbiorow.append(Dane)
            Nazwy_plikow.append(Plik.name)

        Ilosc_plikow = len(Zbior_zbiorow)
        if Ilosc_plikow < 2:
            st.write("")
        else:
            Opcja_wyswietlania = "Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne zbiory"

            Macierz_niepodobienstwa = np.zeros((Ilosc_plikow, Ilosc_plikow))

            for i in range(Ilosc_plikow):
                for j in range(i + 1, Ilosc_plikow):
                    Niepodobienstwo = Niepodobienstwo_Prokrustesa(Zbior_zbiorow[i], Zbior_zbiorow[j])
                    Macierz_niepodobienstwa[i, j] = Niepodobienstwo
                    Macierz_niepodobienstwa[j, i] = Niepodobienstwo

            Macierz_niepodobienstwa_z_nazwami = pd.DataFrame(
                Macierz_niepodobienstwa,
                index=Nazwy_plikow,
                columns=Nazwy_plikow
            )

            if Opcja_wyswietlania == "Pokaż macierz niepodobieństwa":
                st.subheader("Macierz niepodobieństwa Prokrustesa")
                st.dataframe(Macierz_niepodobienstwa_z_nazwami)

            elif Opcja_wyswietlania == "Pokaż macierz niepodobieństwa, oraz znajdź najbardziej podobne zbiory":
                st.subheader("Macierz niepodobieństwa Prokrustesa")
                with st.expander("Wyświetl macierz niepodobieństwa"):
                    st.dataframe(Macierz_niepodobienstwa_z_nazwami)

                Kolumna1, Kolumna2, Kolumna3 = st.columns(3)
                with Kolumna1:
                    st.subheader("Wybierz plik referencyjny")
                    Wybor_zbioru = st.selectbox("Wybierz:", options=Nazwy_plikow)
                with Kolumna2:
                    st.subheader("Podaj liczbę podobnych zbiorów")
                    Liczba_podobnych = st.number_input("Wybierz liczbę najbardziej podobnych zbiorów do odszukwania", min_value=1, max_value=Ilosc_plikow-1, step=1)

                with Kolumna3:
                    st.subheader("Najbardziej podobne zbiory:")
                    if st.button("Znajdź podobne zbiory punktów"):
                        Numer_zbioru = Nazwy_plikow.index(Wybor_zbioru)
                        Niepodobienstwa = Macierz_niepodobienstwa[Numer_zbioru]
                        Indeksy_podobnych = np.argsort(Niepodobienstwa)[1:int(Liczba_podobnych)+1]

                        
                        Podobne_pliki = [Nazwy_plikow[i] for i in Indeksy_podobnych]
                        for i, nazwa_pliku in enumerate(Podobne_pliki):
                            st.write(f"{i + 1}. {nazwa_pliku} (Niepodobieństwo: {Niepodobienstwa[Indeksy_podobnych[i]]:.4f})")

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Wyliczanie odległości Levenshteina pomiędzy słowami
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

elif Opcja_pracy_w_aplikacji == "Wyliczanie odległości Levenshteina pomiędzy ciągami tekstowymi":
    
    Ciag_1 = None
    Ciag_2 = None
    Kolumna1, Kolumna2, Kolumna3 = st.columns(3)
    with Kolumna1:
        st.subheader("Wpisz ciągi tekstowe")
        Ciag_1 = st.text_input("Wpisz pierwszy ciąg:")
        Ciag_2 = st.text_input("Wpisz drugi ciąg:")
    with Kolumna2:
        
        Ciag_1 = Ciag_1.lower()
        Ciag_2 = Ciag_2.lower()

        Ciag_1 = re.sub(r'[^\w\s]', '', Ciag_1)
        Ciag_2 = re.sub(r'[^\w\s]', '', Ciag_2)

        Macierz_programowania_dynamicznego, Odleglosc_Levenshteina_pomiedzy_slowami = Odleglosc_Levenshteina(Ciag_1,Ciag_2)
        st.subheader("Macierz programowania dynamicznego:")
        st.dataframe(pd.DataFrame(Macierz_programowania_dynamicznego))
    with Kolumna3:
        if Ciag_1 != None and Ciag_2 != None:
            st.subheader("Odległość")
            st.write(f"Odległość Levenshteina pomiędzy ciągami wynosi : {Odleglosc_Levenshteina_pomiedzy_slowami}")

###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
# Metryka Czebyszewa na szachownicy
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################

elif Opcja_pracy_w_aplikacji == "Ilość ruchów królem na szachownicy":
    Rozmiar_szachownicy = 8
    Kolumny = [chr(i) for i in range(ord('A'), ord('A') + Rozmiar_szachownicy)]
    Wiersze = list(range(1, Rozmiar_szachownicy + 1))
    Pola = [f"{Kolumna}{Wiersz}" for Kolumna in Kolumny for Wiersz in Wiersze]

    Kolumna1, Kolumna2, Kolumna3 = st.columns(3)

    with Kolumna1:
        st.subheader("Wybierz pola:")
        Pole_startowe = st.selectbox("Wybierz (K):", [None] + Pola)
        Pole_koncowe = st.selectbox("Wybierz (C):", [None] + Pola)

    with Kolumna2:
        if Pole_startowe != None and Pole_koncowe != None:
            Start = Notacja_szachowa_na_wspolrzedne(Pole_startowe)
            Koniec = Notacja_szachowa_na_wspolrzedne(Pole_koncowe)

            Odleglosc_Czebyszewa_na_szachownicy = Odleglosc_Czebyszewa(Start, Koniec)

            Plansza = np.zeros((Rozmiar_szachownicy, Rozmiar_szachownicy), dtype=str)
            Plansza[Start[0], Start[1]] = "K"
            Plansza[Koniec[0], Koniec[1]] = "C"

            Plansza_do_wyswietlenia = pd.DataFrame(
                Plansza,
                index=[str(i) for i in range(Rozmiar_szachownicy, 0, -1)],
                columns=Kolumny
            )

            st.subheader("Szachownica wraz z ustawieniem:")
            st.dataframe(Plansza_do_wyswietlenia.style.apply(lambda x: Styl_szachownicy(Plansza_do_wyswietlenia), axis=None))
    with Kolumna3:
        if Pole_startowe != None and Pole_koncowe != None:    
            st.subheader("Wynik:")
            st.write(f"Odległość Czebyszewa między polami {Pole_startowe} i {Pole_koncowe} wynosi: {Odleglosc_Czebyszewa_na_szachownicy}")

    st.subheader("")
    st.subheader("")
    st.subheader("")
    st.subheader("")
    st.subheader("")



###################################################################################################################################
###################################################################################################################################
###################################################################################################################################
####### Wykrywanie wartości odstających w zbiorze danych: dane w formacie CSV/XLS/XLSX
###################################################################################################################################
###################################################################################################################################
###################################################################################################################################


elif Opcja_pracy_w_aplikacji == "Wykrywanie wartości odstających w zbiorze danych: dane w formacie CSV/XLS/XLSX":
    Plik_z_danymi = None
    st.subheader("Czy chcesz przetestować funkcje na swoich danych?")

    Wybor_wygrywania_plikow = st.selectbox("Wybierz:", [None, "Korzystaj z plików aplikacji","Wgraj własne dane"])
    
    Kolumna1OB, Kolumna2OB = st.columns(2)

    if Wybor_wygrywania_plikow == "Korzystaj z plików aplikacji":
        with Kolumna1OB:
            st.subheader("Wybierz zbiór dostępnych danych")
            st.write("")
            with st.expander("Dostępne zbiory"):
                st.write("- Dane dwuwymiarowe")
                st.write("- Dane z dwuwymiarowego rozkładu normalnego")
                st.write("- Dane trzywymiarowe")
                st.write("- Dane z trzywymiarowego rozkładu normalnego")
        with Kolumna2OB:
            st.subheader("Lista wyboru")
            Rodzaj_danych = st.selectbox("Wybierz:",[None, "Dane dwuwymiarowe","Dane z dwuwymiarowego rozkładu normalnego","Dane trzywymiarowe","Dane z trzywymiarowego rozkładu normalnego"])

        if Rodzaj_danych == "Dane z dwuwymiarowego rozkładu normalnego":
            Folder_pliki = r"Outliers_2D_normal"
            Plik_csv = next((f for f in os.listdir(Folder_pliki) if f.endswith('.csv')), None)

            if Plik_csv:
                Sciezka_pliku = os.path.join(Folder_pliki, Plik_csv)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv

        elif Rodzaj_danych == "Dane dwuwymiarowe":
            Folder_pliki = r"Outliers_2D"
            Plik_csv = next((f for f in os.listdir(Folder_pliki) if f.endswith('.csv')), None)

            if Plik_csv:
                Sciezka_pliku = os.path.join(Folder_pliki, Plik_csv)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv

        elif Rodzaj_danych == "Dane trzywymiarowe":
            Folder_pliki = r"Outliers_3D"
            Plik_csv = next((f for f in os.listdir(Folder_pliki) if f.endswith('.csv')), None)

            if Plik_csv:
                Sciezka_pliku = os.path.join(Folder_pliki, Plik_csv)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv

        elif Rodzaj_danych == "Dane z trzywymiarowego rozkładu normalnego":
            Folder_pliki = r"Outliers_3D_normal"
            Plik_csv = next((f for f in os.listdir(Folder_pliki) if f.endswith('.csv')), None)

            if Plik_csv:
                Sciezka_pliku = os.path.join(Folder_pliki, Plik_csv)
                with open(Sciezka_pliku, "rb") as f:
                    Plik_z_danymi = BytesIO(f.read())
                    Plik_z_danymi.name = Plik_csv


    elif Wybor_wygrywania_plikow == "Wgraj własne dane":
        st.subheader("Wgraj plik w formacie CSV/XLS/XLSX")

        Plik_z_danymi = st.file_uploader("Wgraj plik z danymi w formacie CSV/XLS/XLSX", type=["csv", "xls", "xlsx"])

    if Wybor_wygrywania_plikow != None and Plik_z_danymi != None:
        if Plik_z_danymi.name.endswith(('xls', 'xlsx')):
            Dane = pd.read_excel(Plik_z_danymi)
        else:
            Dane = pd.read_csv(Plik_z_danymi)
        st.subheader("Opcjonalny podgląd danych")
        with st.expander("Podgląd wybranego zbioru danych:"):
            st.dataframe(Dane)

        Dane = Dane.to_numpy()

        Wymiary = Dane.shape[1]

        Wartosc_alpha = 0.05

        Statystyka, p, _ = multivariate_normality(Dane, alpha=Wartosc_alpha)

        st.subheader("Analiza normalności zbioru danych")
        if p > Wartosc_alpha:
            st.write(f"Dane mogą pochodzić z {Wymiary}-wymiarowego rozkładu normalnego.")
            st.subheader("Wybierz miarę niepodobieństwa")
            with st.expander("Dostępne miary:"):
                st.write("- Metryka Euklidesa")
                st.write("- Metryka Mahalanobisa")
            Wybor_miary = st.selectbox("Wybierz:",[None, "Metryka Euklidesa","Metryka Mahalanobisa"])

        else:
            st.write(f"Odrzucamy hipotezę o normalności danych.")
            st.subheader("Wybierz miarę niepodobieństwa")
            with st.expander("Dostępne miary:"):
                st.write("- Metryka Manahattan")
                st.write("- Metryka Euklidesa")
                st.write("- Metryka Czebyszewa")
            Wybor_miary = st.selectbox("Wybierz:",[None, "Metryka Manhattan","Metryka Euklidesa","Metryka Czebyszewa"])
            
        if Wybor_miary == "Metryka Mahalanobisa":
            Wektor_srednich = np.mean(Dane, axis=0)
            Macierz_kowariancji_danych = np.cov(Dane, rowvar=False, ddof=1)
            Odwrotnosc_macierzy_kowariancji_danych = np.linalg.inv(Macierz_kowariancji_danych)

            Kolumna010, Kolumna020 = st.columns(2)

            with Kolumna010:
                st.subheader(rf'''Wybierz prawdopodobieństwo $1 - \alpha$:''')
                Prawdopodobienstwo_Mahalanobis = st.slider(
                    "Ustaw wartość prawdopodobieństwa:",
                    min_value=0.8,
                    max_value=0.999,
                    value=0.95,
                    step=0.001
                )

                Wymiary = Dane.shape[1]
                Kwantyl_chi2 = chi2.ppf(Prawdopodobienstwo_Mahalanobis, df=Wymiary)

                with st.expander("Wyświetl szczegóły"):
                    st.markdown(rf'''Kwadrat odległości Mahalanobisa od wektora średnich: $d^2_M(X_k,\mu)$, jest miarą tego jak bardzo obserwacja odbiega od wektora średnich $\mu$''')
                    st.markdown(rf'''Wybrana wartość $1-\alpha$ będzie określać prawdopodobieństwo że obserwacja odbiega od wektora średnich $\mu$, zgodnie z równaniem:''')
                    st.markdown(rf'''$$P(d^2_M(X_k,\mu) \leqslant \chi^2_{{n, 1-\alpha}}) = 1 - \alpha$$''')
                    st.markdown(rf'''To znaczy:''')
                    st.markdown(rf'''$$P(d^2_M(X_k,\mu) \leqslant \chi^2_{{{Wymiary}, {Prawdopodobienstwo_Mahalanobis}}}) = {Prawdopodobienstwo_Mahalanobis}$$''')
                    st.markdown(rf'''Wartość krytyczna rozkładu: $$\chi^2_{{{Wymiary}, {Prawdopodobienstwo_Mahalanobis}}}$$ = {Kwantyl_chi2}''')
                    st.markdown(rf'''Aplikacja identyfikuje obserwacje jako nietypowe w modelu, jeżeli nie spełniają nierówności: ''')
                    st.markdown(rf'''$d_M(X_k,\mu) \leqslant ${np.sqrt(Kwantyl_chi2)}''')
                    st.markdown(rf'''To znaczy jeżeli:''')
                    st.markdown(rf'''$d^2_M(X_k,\mu) > \chi^2_{{n, 1-\alpha}}$''')

            
            

            Odleglosci_Mahalanobisa = [
                Odleglosc_Mahalanobisa(x, Wektor_srednich, Odwrotnosc_macierzy_kowariancji_danych) for x in Dane
            ]
            Kwadraty_odleglosci_Mahalanobisa = np.array(Odleglosci_Mahalanobisa) ** 2

            Wartosci_odstajace = np.where(Kwadraty_odleglosci_Mahalanobisa > Kwantyl_chi2)[0]

            with Kolumna020:
                if len(Wartosci_odstajace) > 0:
                    st.subheader(f"Liczba wartości nietypowych: {len(Wartosci_odstajace)}")
                    Wyswietlanie_wartosci_odstajacych = st.radio("Czy chcesz wyświetlić szczegółowe informacje o nietypowych obserwacjach?",["Nie","Tak"])
                    if Wyswietlanie_wartosci_odstajacych == "Tak":
                        st.write("Wykryte wartości nietypowe:")
                        for idx in Wartosci_odstajace:
                            st.write(f"Indeks: {idx}, Odległość Mahalanobisa: {Odleglosci_Mahalanobisa[idx]:.4f}")
                    
                else:
                    st.write("Nie wykryto wartości odstających.")

            

            Wartosci_wlasne, Wektory_wlasne, Lambdy, Elipsoida_obrocona_i_przesunieta = Narysuj_punkty_i_granice_decyzyjne_odleglosci_Mahalanobisa(Dane, Wektor_srednich, Macierz_kowariancji_danych, Prawdopodobienstwo_Mahalanobis)

            if Wymiary == 2:
                Wykres = go.Figure()
                Wykres.add_trace(go.Scatter(x = Dane[:, 0], y = Dane[:, 1],
                    mode='markers', marker=dict(size=8, color=Odleglosci_Mahalanobisa, colorscale='Viridis', colorbar=dict(title='Odległość Mahalanobisa'), opacity=0.8)))

                Wykres.add_trace(go.Scatter(x = [Wektor_srednich[0]], y = [Wektor_srednich[1]],
                    mode='markers', marker=dict(size=12, color='red', symbol='x')))

                Wykres.add_trace(go.Scatter(x = Elipsoida_obrocona_i_przesunieta[0], y = Elipsoida_obrocona_i_przesunieta[1],
                    mode='lines', line=dict(color='orange', width=2)))

                for i in range(2):
                    Wektor_wlasny = Wektory_wlasne[:, i] * np.sqrt(Wartosci_wlasne[i]) * np.sqrt(Kwantyl_chi2)
                    Wykres.add_trace(go.Scatter(
                        x=[Wektor_srednich[0], Wektor_srednich[0] + Wektor_wlasny[0]],
                        y=[Wektor_srednich[1], Wektor_srednich[1] + Wektor_wlasny[1]],
                        mode='lines+markers', line=dict(color='orange', width=2, dash='dash')))

                Wykres.update_layout(width=1200, height=600, showlegend=False)

                st.plotly_chart(Wykres, use_container_width=False)

            elif Wymiary == 3:
                Wykres = go.Figure()
                Wykres.add_trace(go.Scatter3d(x = Dane[:, 0], y = Dane[:, 1], z = Dane[:, 2],
                    mode='markers', marker=dict(size=5, color=Odleglosci_Mahalanobisa, colorscale='Viridis', colorbar=dict(title='Odległość Mahalanobisa')), name='Punkty danych'))

                Wykres.add_trace(go.Scatter3d(x = [Wektor_srednich[0]], y = [Wektor_srednich[1]], z = [Wektor_srednich[2]],
                    mode='markers', marker=dict(size=10, color='red', symbol='cross'), name="Wektor średnich"))

                Wykres.add_trace(go.Surface(x = Elipsoida_obrocona_i_przesunieta[0],y = Elipsoida_obrocona_i_przesunieta[1],z = Elipsoida_obrocona_i_przesunieta[2],
                    colorscale='Oranges', opacity=0.1, showscale=False, name='Granica decyzyjna'))

                for i in range(3):
                    Wektor_wlasny = Wektory_wlasne[:, i] * np.sqrt(Wartosci_wlasne[i]) * np.sqrt(Kwantyl_chi2)
                    Wykres.add_trace(go.Scatter3d(x = [Wektor_srednich[0], Wektor_srednich[0] + Wektor_wlasny[0]], y = [Wektor_srednich[1], Wektor_srednich[1] + Wektor_wlasny[1]], z = [Wektor_srednich[2], Wektor_srednich[2] + Wektor_wlasny[2]],
                        mode='lines', line=dict(color='orange', width=5, dash='dash')))

                Wykres.update_layout(width=1400, height=1000, showlegend = False)

                st.plotly_chart(Wykres, use_container_width=False)

            elif Wymiary != 2 or Wymiary != 3:
                st.write("Wizualizacja dostępna dla danych o 2 lub 3 wymiarach.")

        elif Wybor_miary == "Metryka Manhattan" or Wybor_miary == "Metryka Euklidesa" or Wybor_miary == "Metryka Czebyszewa":
            Wektor_srednich = np.mean(Dane, axis=0)

            st.subheader("Wybierz percentyl jako próg odcięcia:")
            with st.expander("Wyświetl szczegóły"):
                st.markdown(rf'''Wybrany percentyl będzie określał ile procent obserwacji ma zostać zidentyfikowane jako nietypowe w modelu''')
            Percentyl = st.slider(
                "Ustaw percentyl:",
                min_value=50.0,
                max_value=99.9,
                value=95.0,
                step=0.1
            )
            st.write(f"Percentyl: {Percentyl:.2f}")

            Wymiary = Dane.shape[1]

            if Wybor_miary == "Metryka Manhattan":
                Odleglosci = [Odleglosc_Manhattan(punkt, Wektor_srednich) for punkt in Dane]
            elif Wybor_miary == "Metryka Euklidesa":
                Odleglosci = [Odleglosc_Euklidesa(punkt, Wektor_srednich) for punkt in Dane]
            elif Wybor_miary == "Metryka Czebyszewa":
                Odleglosci = [Odleglosc_Czebyszewa(punkt, Wektor_srednich) for punkt in Dane]
            if Percentyl:
                Odleglosc_odciecia = np.percentile(Odleglosci, float(Percentyl))
                st.markdown(rf'''Odległość odcięcia powyżej której obserwacje są uznawane za nietypowe = {Odleglosc_odciecia}''')
                Obserwacje_odstajace = np.where(Odleglosci > Odleglosc_odciecia)[0]

                if len(Obserwacje_odstajace) > 0:
                    st.subheader(f"Liczba wartości nietypowych: {len(Obserwacje_odstajace)}")
                    Wyswietlanie_wartosci_odstajacych = st.radio("Czy chcesz wyświetlić szczegółowe informacje o nietypowych obserwacjach?",["Nie","Tak"])
                    if Wyswietlanie_wartosci_odstajacych == "Tak":
                        st.write("Wykryte wartości nietypowe:")
                        for idx in Obserwacje_odstajace:
                            st.write(f"Indeks: {idx}, Odległość od centrum: {Odleglosci[idx]:.4f}")
                
            else:
                st.write("Nie wykryto wartości odstających.")

            if Wymiary == 2:
                Wykres = go.Figure()
                Wykres.add_trace(go.Scatter(x = Dane[:, 0], y= Dane[:, 1],
                    mode='markers', marker=dict(size=8,color=Odleglosci, colorscale='Viridis', colorbar=dict(title='Odległość'), opacity=0.8), name='Punkty danych'))
                Wykres.add_trace(go.Scatter(x = [Wektor_srednich[0]], y = [Wektor_srednich[1]],
                    mode='markers', marker=dict(size=12, color='red', symbol='x')))


                if Wybor_miary == "Metryka Euklidesa":
                    Okrag_x = Wektor_srednich[0] + Odleglosc_odciecia * np.cos(np.linspace(0, 2 * np.pi, 500))
                    Okrag_y = Wektor_srednich[1] + Odleglosc_odciecia * np.sin(np.linspace(0, 2 * np.pi, 500))
                    Wykres.add_trace(go.Scatter(x = Okrag_x, y = Okrag_y,
                        mode='lines', line=dict(color='orange', dash='dash')))
                elif Wybor_miary == "Metryka Manhattan":
                    Romb_x = [Wektor_srednich[0] + Odleglosc_odciecia, Wektor_srednich[0], Wektor_srednich[0] - Odleglosc_odciecia, Wektor_srednich[0], Wektor_srednich[0] + Odleglosc_odciecia]
                    Romb_y = [Wektor_srednich[1], Wektor_srednich[1] + Odleglosc_odciecia, Wektor_srednich[1], Wektor_srednich[1] - Odleglosc_odciecia, Wektor_srednich[1]]
                    Wykres.add_trace(go.Scatter(x = Romb_x, y = Romb_y,
                        mode='lines', line=dict(color='orange', dash='dash')))
                elif Wybor_miary == "Metryka Czebyszewa":
                    Kwadrat_x = [Wektor_srednich[0] + Odleglosc_odciecia, Wektor_srednich[0] + Odleglosc_odciecia, Wektor_srednich[0] - Odleglosc_odciecia, Wektor_srednich[0] - Odleglosc_odciecia, Wektor_srednich[0] + Odleglosc_odciecia]
                    Kwadrat_y = [Wektor_srednich[1] + Odleglosc_odciecia, Wektor_srednich[1] - Odleglosc_odciecia, Wektor_srednich[1] - Odleglosc_odciecia, Wektor_srednich[1] + Odleglosc_odciecia, Wektor_srednich[1] + Odleglosc_odciecia]
                    Wykres.add_trace(go.Scatter(x = Kwadrat_x, y = Kwadrat_y,
                        mode='lines', line=dict(color='orange', dash='dash')))

                Wykres.update_layout(width=1200, height=800, showlegend=False)

                st.plotly_chart(Wykres, use_container_width=False)


            if Wymiary == 3:
                Wykres3D = go.Figure()
                Wykres3D.add_trace(go.Scatter3d(x = Dane[:, 0], y = Dane[:, 1], z = Dane[:, 2],
                    mode='markers',marker=dict(size=5, color=Odleglosci, colorscale='Viridis', colorbar=dict(title='Odległość'), opacity=0.8)))

                Wykres3D.add_trace(go.Scatter3d(x = [Wektor_srednich[0]], y = [Wektor_srednich[1]], z = [Wektor_srednich[2]],
                    mode='markers', marker=dict(size=10, color='red', symbol='cross'), name="Wektor średnich"))

                if Wybor_miary == "Metryka Euklidesa":
                    U, V = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100))
                    
                    Wykres3D.add_trace(go.Surface(x = Wektor_srednich[0] + Odleglosc_odciecia * np.sin(V) * np.cos(U), y = Wektor_srednich[1] + Odleglosc_odciecia * np.sin(V) * np.sin(U), z = Wektor_srednich[2] + Odleglosc_odciecia * np.cos(V),
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))

                elif Wybor_miary == "Metryka Manhattan":
                    U, V = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))

                    Z = Odleglosc_odciecia - np.abs(U * Odleglosc_odciecia) - np.abs(V * Odleglosc_odciecia)
                    Z_poz = Wektor_srednich[2] + Z
                    Z_neg = - Z_poz

                    Maska = (np.abs(U * Odleglosc_odciecia) + np.abs(V * Odleglosc_odciecia) <= Odleglosc_odciecia)

                    X = Wektor_srednich[0] + U * Odleglosc_odciecia
                    Y = Wektor_srednich[1] + V * Odleglosc_odciecia

                    Z_poz = np.where(Maska, Wektor_srednich[2] + Z_poz, np.nan)
                    Z_neg = np.where(Maska, Wektor_srednich[2] + Z_neg, np.nan)

                    Wykres3D.add_trace(go.Surface(x = X, y = Y, z = Z_poz,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))

                    Wykres3D.add_trace(go.Surface(x = X, y = Y, z = Z_neg,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))

                elif Wybor_miary == "Metryka Czebyszewa":
                    X, Y = np.meshgrid(np.linspace(Wektor_srednich[0] - Odleglosc_odciecia, Wektor_srednich[0] + Odleglosc_odciecia, 50), np.linspace(Wektor_srednich[1] - Odleglosc_odciecia, Wektor_srednich[1] + Odleglosc_odciecia, 50))

                    Z1 = np.full_like(X, Wektor_srednich[2] + Odleglosc_odciecia)
                    Z2 = np.full_like(X, Wektor_srednich[2] - Odleglosc_odciecia)

                    Wykres3D.add_trace(go.Surface(x = X, y = Y, z = Z1,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))
                    Wykres3D.add_trace(go.Surface(x = X, y = Y, z = Z2,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))

                    Y, Z = np.meshgrid(np.linspace(Wektor_srednich[1] - Odleglosc_odciecia, Wektor_srednich[1] + Odleglosc_odciecia, 50), np.linspace(Wektor_srednich[2] - Odleglosc_odciecia, Wektor_srednich[2] + Odleglosc_odciecia, 50))

                    X1 = np.full_like(Y, Wektor_srednich[0] + Odleglosc_odciecia)
                    X2 = np.full_like(Y, Wektor_srednich[0] - Odleglosc_odciecia)

                    Wykres3D.add_trace(go.Surface(x = X1, y = Y, z = Z,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))
                    Wykres3D.add_trace(go.Surface(x = X2, y = Y, z = Z,
                        opacity=0.1, colorscale='Oranges', showscale=False, name="Granica decyzyjna"))

                Wykres3D.update_layout(width=1200, height=800, showlegend=False)

                st.plotly_chart(Wykres3D, use_container_width=False)


            elif Wymiary != 2 or Wymiary != 3:
                st.write("Wizualizacja dostępna dla danych o 2 lub 3 wymiarach.")




            








