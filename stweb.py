import streamlit as st
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import plot_confusion_matrix,plot_precision_recall_curve,plot_roc_curve
from sklearn.metrics import precision_score,recall_score 
def main():
    st.title("welcome to web development 🏏")
    st.sidebar.title("welcome to web development")
    st.markdown("here we are using python  🏏")
    st.sidebar.title("here we are using python  🏏")

    @st.cache(persist=True)
    def load_data():
        data=pd.read_csv('C:/Users/ANUPAMREDDY/Desktop/mushrooms.csv')
        label=LabelEncoder()
        for col in data.columns:
            data[col]=label.fit_transform(data[col])
        return data
    
    @st.cache(persist=True)
    def split(df):
        y=df.type
        x=df.drop(columns=['type'])
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)
        return x_train,x_test,y_train,y_test

    def plot_metrics(metrics_list):
        if 'confusion matrix' in metrics_list:
            st.subheader("confusion matrix")
            plot_confusion_matrix(model,x_test,y_test,display_labels=class_names)    
            st.pyplot()

        if 'roc curve' in metrics_list:
            st.subheader("roc curve")
            plot_roc_curve(model,x_test,y_test)    
            st.pyplot() 
        if 'precision-recall curve' in metrics_list:
            st.subheader("precision-recall curve")
            plot_precision_recall_curve(model,x_test,y_test)    
            st.pyplot()    

    df=load_data()
    x_train,x_test,y_train,y_test=split(df)
    class_names=['edible','poisonous']
    st.sidebar.subheader("choose classifier")
    classifier = st.sidebar.selectbox("classifier",("svm","logistic regression","random forest"))

    if classifier == 'svm':
        st.sidebar.subheader("model hyperparameters")
        c=st.sidebar.number_input("c regularisation parameter",0.01,10.0,step=0.01,key="c")
        kernel=st.sidebar.radio("kernel",("rbf","linear"),key="kernel")
        gamma=st.sidebar.radio("gamma",("scale","auto"),key='gamma')

        metrics = st.sidebar.multiselect("what metrics to plot?", ('confusion matrix','roc curve','precision-recall curve'))

        if st.sidebar.button("classify",key='classify'):
            st.subheader("svm results")
            model =SVC(C=c, kernel=kernel, gamma=gamma)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("accuracy: ",accuracy.round(2))
            st.write("precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'logistic regression':
        st.sidebar.subheader("model hyperparameters")
        c=st.sidebar.number_input("c regularisation parameter",0.01,10.0,step=0.01,key="c_lr")
        max_iter=st.sidebar.slider("maximum number of iterations",100,500,key="max_iter")

        metrics = st.sidebar.multiselect("what metrics to plot?", ('confusion matrix','roc curve','precision-recall curve'))

        if st.sidebar.button("classify",key='classify'):
            st.subheader("logistic regression results")
            model =LogisticRegression(C=c,max_iter=max_iter)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("accuracy: ",accuracy.round(2))
            st.write("precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)

    if classifier == 'random forest':
        st.sidebar.subheader("model hyperparameters")
        n_estimators=st.sidebar.number_input("number of trees",100,5000,step=10,key="n_estimators")
        max_depth=st.sidebar.number_input("max depth of forest",3,20,key="max_depth")
        bootstrap=st.sidebar.radio("bootstrap value",("true","false"),key="bootstrap")
        metrics = st.sidebar.multiselect("what metrics to plot?", ('confusion matrix','roc curve','precision-recall curve'))

        if st.sidebar.button("classify",key='classify'):
            st.subheader("random forest results")
            model =RandomForestClassifier(n_estimators=n_estimators,max_depth=max_depth,bootstrap=bootstrap,n_jobs=-1)
            model.fit(x_train,y_train)
            accuracy=model.score(x_test,y_test)
            y_pred=model.predict(x_test)
            st.write("accuracy: ",accuracy.round(2))
            st.write("precision: ",precision_score(y_test,y_pred,labels=class_names).round(2))
            st.write("recall: ",recall_score(y_test,y_pred,labels=class_names).round(2))
            plot_metrics(metrics)            

    if st.sidebar.checkbox("show the data",False):
        st.subheader("mushroom data set")
        st.write(df)        
if __name__ == '__main__':
    main()
     