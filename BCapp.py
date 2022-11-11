"""  Box counting algorithm for estimating fractal dimension of a binarized image.

Adapted from

https://www.physics.utoronto.ca/apl/fvf/python_code/box_count.py
 and
https://francescoturci.wordpress.com/2016/03/31/box-counting-in-numpy/

"""

import pylab as pl
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

#############################################

def CarregaImagem(NomeArq):
    image_file_name = NomeArq # should be a tif file, but could be anything
    #
    image = Image.open(image_file_name)  # create PIL image object
    image_grayscale = image.convert('L')   # cast the image into grey scale 0-255 (just in case)
    #
    image_matrix = np.asmatrix(image_grayscale).copy()  # copy the greyscale values into numpy matrix format
    #
    # binarize the image (it may be thresholded already)
    thresh = 200  # 0 is black 255 is white
    #
    for i in range(image.size[1]):
        for j in range(image.size[0]):
            if image_matrix[i,j] > thresh:
                image_matrix[i,j] = 255    # white
            else:
                image_matrix[i,j] = 0   # black
    #
    img = Image.fromarray(image_matrix)      	# save the binarized image
    img.save('fig_binarized.tif')  # as a check against the original image
    img2 = Image.open('fig_binarized.tif')
    st.image(img2, caption='Figura "binarizada"(PB)')
    return image_matrix


####
def BoxCounting(image_matrix):
    # Make a list of black pixel coordinates
    pixels=[]
    for i in range(image_matrix.shape[0]):
        for j in range(image_matrix.shape[1]):
            if image_matrix[i,j] == 0:    #  pixel is black
                pixels.append((i,j))      #  count it
    #            
    Lx=image_matrix.shape[0]
    Ly=image_matrix.shape[1]
    #print(Lx,Ly)
    #
    pixels=pl.array(pixels)   # turn into a pylab array

    # computing the fractal dimension
    # considering only scales in a logarithmic list
    scales = 1 + np.array([2, 3, 5, 8, 13, 21, 34, 55, 89, 144])
    #
    #scales = 1 + np.array([ 5, 8, 13, 21, 34, 55, 89, 144])
    #
    Ns  = []  # number of squares
    Ndiv= []  # number of divisions
    # looping over several scales
    for scale in scales:
        if Lx/scale < 2:
            continue
        #
        #print('======= Scale :', scale)
        # computing the histogram
        gridX, gridY = np.linspace(0,Lx-1,scale), np.linspace(0,Ly-1,scale)
        H, edges=np.histogramdd(pixels, bins=( gridX, gridY ))
        Ns.append(np.sum(H > 0))
        Ndiv.append(len(gridY)-1)
    #
    return Ns, Ndiv
    
################################    
# Função principal da aplicação
def main():
    # Configuração inicial da página da aplicação
    st.set_page_config(page_title = 'Box Counting / Df', \
        layout="wide",
        initial_sidebar_state='expanded'
    )
    # Na barra lateral:
    latexto = r''' Dimensão fractal $D_f$ de imagens'''
    st.sidebar.write(latexto)
    st.sidebar.markdown("---")
    #
    uploaded_file = st.sidebar.file_uploader('Escolha uma imagem (que deve ser "quadrada"):')
    #
    if uploaded_file is None:
        new_title = '<p style="font-family:sans-serif; color:Green; font-size: 30px;"> Estimando a dimensão fractal de imagens </p>'
        st.markdown(new_title, unsafe_allow_html=True)
        st.markdown("***")
        st.write("<= Escolha OU arraste um arquivo de imagem")
    else:
        Cont1 = st.container()
        # Colunas:
        col1, col2 = Cont1.columns(2)  
        with col1:
            imagem = Image.open(uploaded_file)
            st.image(imagem, caption='Figura carregada')
        with col2:
            NomeArq = uploaded_file
            Matrix = CarregaImagem(NomeArq)
            Ns, Ndiv = BoxCounting(Matrix)
            # linear fit, polynomial of degree 1 --> a line
            coeffs, covar = np.polyfit(np.log(Ndiv), np.log(Ns), 1, cov=True)
        #
        Cont2 = st.container()
        Cont2.markdown("---")
        D = coeffs[0]  #the fractal dimension is the slope of the line
        Cont2.markdown('#### Estimativa por "*box counting*"(BC):')
        textoD = r''' $D_f \approx$ '''+str(D)
        Cont2.write(textoD)
        erro = np.sqrt(np.diag(covar)[0])
        if erro > 1.0e-6:
            textoE = r''' erro $\approx$ '''+str( erro )
            Cont2.write(textoE)
        #
        Cont3 = st.container()
        Mostrar = Cont3.checkbox('Mostrar dados do BC e ajuste')
        if Mostrar:
            fig, ax = plt.subplots(figsize=(5,4))
            ax.plot(np.log(Ndiv), np.log(Ns), 'o', mfc='none', label='dados')
            ax.plot(np.log(Ndiv), np.polyval(coeffs, np.log(Ndiv)),label='ajuste')
            ax.set_xlabel('ln(Ndivs)')
            ax.set_ylabel('ln(Nquadrados)')
            ax.legend()
            fig.savefig("figure2.png")
            image2 = Image.open('figure2.png')
            Cont3.image(image2)
                    
##########################################################
if __name__ == '__main__':
	main()
