import streamlit as st
import pandas as pd

st.write('# Two Seven')

df_two_seven = pd.read_csv("two_seven_clean.csv").dropna()

lr = st.selectbox("Learning Rate", df_two_seven["Learning Rate"].unique())
epoch = st.selectbox("Epoch", df_two_seven[" Epoch"].unique())
batch_sizes = st.selectbox("Batch Size", df_two_seven["Batch Size"].unique())
pca_components = st.selectbox("PCA Components", df_two_seven["PCA Comp"].unique())
normalization = st.selectbox("Normalization",df_two_seven["Norm"].unique())


accuracy = df_two_seven[(df_two_seven["Learning Rate"]==lr)&(df_two_seven[" Epoch"]==epoch)&(df_two_seven["Batch Size"]==batch_sizes)&(df_two_seven['PCA Comp']==pca_components)&(df_two_seven['Norm']==normalization)]["Loss"].iloc[0]
st.write("Accuracy: ", round(accuracy*100,3),"%")

st.write('# Eight five')

df1 = pd.read_csv("eight_five-2.csv").dropna()

lr1= st.selectbox("Learning Rate ", df1["Learning Rate"].unique())
epoch1 = st.selectbox("Epoch ", df1[" Epoch"].unique())
batch_sizes1 = st.selectbox("Batch Size ", df1["Batch Size"].unique())
pca_components1 = st.selectbox("PCA Components ", df1["PCA Comp"].unique())
normalization1 = st.selectbox("Normalization ",df1["Norm"].unique())


accuracy1 = df1[(df1["Learning Rate"]==lr1)&(df1[" Epoch"]==epoch1)&(df1["Batch Size"]==batch_sizes1)&(df1['PCA Comp']==pca_components1)&(df1['Norm']==normalization1)]["Loss"].iloc[0]
st.write("Accuracy:  ", round(accuracy1*100,3),"%")

st.write('# All Digits ')

df2 = pd.read_csv("softmax_clean.csv").dropna()

lr2 = st.selectbox("Learning Rate  ", df2["Learning Rate"].unique())
epoch2 = st.selectbox("Epoch  ", df2[" Epoch"].unique())
batch_sizes2 = st.selectbox("Batch Size  ", df2["Batch Size"].unique())
pca_components2 = st.selectbox("PCA Components  ", df2["PCA Comp"].unique())
normalization2 = st.selectbox("Normalization  ",df2["Norm"].unique())


accuracy2 = df2[(df2["Learning Rate"]==lr2)&(df2[" Epoch"]==epoch2)&(df2["Batch Size"]==batch_sizes2)&(df2['PCA Comp']==pca_components2)&(df2['Norm']==normalization2)]["Loss"].iloc[0]
st.write("Accuracy:   ", round(accuracy2*100,3),"%")